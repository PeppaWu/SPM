from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pointnet2_ops import pointnet2_utils

try:
    from mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .block import Block
from .build import MODELS, build_spike_node

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, timestep, spike_mode):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 128, 1),
            nn.BatchNorm2d(128),
            build_spike_node(timestep, spike_mode),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
        )
        self.second_conv = nn.Sequential(
            build_spike_node(timestep, spike_mode),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            build_spike_node(timestep, spike_mode),
            nn.Conv2d(512, self.encoder_channel, 1),
            nn.BatchNorm2d(self.encoder_channel),
        )

    def forward(self, point_groups):
        '''
            point_groups : T B G N 3
            -----------------
            feature_global : T B G C
        '''  
        ts, bs, g, n, _ = point_groups.shape
        point_groups = point_groups.flatten(0,1)
        # encoder
        feature = self.first_conv(point_groups.permute(0,3,1,2))  # TB 256 G n  
        feature_global = torch.max(feature, dim=3, keepdim=True)[0] # TB 256 G 1
        feature = torch.cat([feature_global.expand(-1, -1, -1, n), feature], dim=1)  # TB 512 G n
        feature = self.second_conv(feature)  # TB 384 G n
        feature_global = torch.max(feature, dim=3, keepdim=False)[0] # TB 384 G 
        return feature_global.transpose(-1,-2)  


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, expand, timestep):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.T = timestep
        self.expand = expand
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, data):
        '''
            input: T B N 3
            ---------------------------
            output: T B G M 3
            center : T B G 3
        '''
        xyz = data[..., :3]
        _, batch_size, num_points, _ = xyz.shape
        step_size_f = int((self.expand - 1.0)*self.num_group/self.T*2)
        step_size_b = int((self.expand - 1.0)*self.num_group)
        xyz = xyz.flatten(0,1)

        # fps the centers out
        # center = misc.fps(xyz, self.num_group)  # B G 3                  # NOT Moving
        # center = misc.fps(xyz, int(self.num_group*self.expand))  # B G' 3 # NOT Moving BUT Expanding
        # F:Whole Moving
        # center = misc.fps(xyz[:batch_size], self.num_group + step_size_f*self.T) # B G' 3
        # center = torch.stack([center[:, (i * step_size_f):(i * step_size_f + self.num_group)] 
        #                              for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3
        # B:Sampling
        # center = misc.fps(xyz[:batch_size], self.num_group + step_size_b*(self.T-1)) # B G' 3
        # center = torch.stack([torch.cat((center[:, :(self.num_group - step_size_b)], \
        #                     center[:, ((i-1) * step_size_b + self.num_group): \
        #                     (i * step_size_b + self.num_group)]), dim=1) \
        #                     for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3     
        # F:Whole Moving B:Sampling                                                                              
        center_idx = pointnet2_utils.furthest_point_sample(xyz[:batch_size].contiguous(), self.num_group + (step_size_f+step_size_b)*(self.T-1)) # B G' 3 
        center = pointnet2_utils.gather_operation(data[0].transpose(1, 2).contiguous(), center_idx).transpose(1, 2).contiguous()   
        center = torch.stack([torch.cat((center[:, i*step_size_f : i*step_size_f + (self.num_group - step_size_b)], \
                            center[:, ((i-1) * step_size_b + self.num_group + (self.T-1)*step_size_f):\
                            (i * step_size_b + self.num_group + (self.T-1)*step_size_f)]), dim=1) \
                            for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3       
        # F:Part Moving B:Sampling
        # center = misc.fps(xyz[:batch_size], self.num_group + (step_size_f+step_size_b)*(self.T-1)) # B G' 3  
        # center = torch.stack([torch.cat((center[:, i*step_size_f:(i+1)*step_size_f], \
        #                     center[:, self.T*step_size_f : (self.num_group - step_size_b + (self.T-1)*step_size_f)], \
        #                     center[:, ((i-1) * step_size_b + self.num_group + (self.T-1)*step_size_f):\
        #                     (i * step_size_b + self.num_group + (self.T-1)*step_size_f)]), dim=1) \
        #                     for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3  

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center[..., :3])  # TB G M
        idx_base = torch.arange(0, batch_size * self.T, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = data.view(self.T * batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(self.T, batch_size, self.num_group, self.group_size, -1).contiguous()
        center = center.view(self.T, batch_size, self.num_group, 1, -1).contiguous()
        # normalize
        neighborhood[..., :3] -= center[..., :3]
        return neighborhood, center.squeeze(-2)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        timestep=2,
        spike_mode="lif",
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if spike_mode is None:
        from mamba_ssm.modules.mamba_simple import Mamba # ANN Mamba Block
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        from mamba.mamba_ssm.modules.mamba_simple import Mamba # SNN Mamba Block
        mixer_cls = partial(Mamba, timestep=timestep, spike_mode=spike_mode, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.BatchNorm1d if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        timestep,
        spike_mode=spike_mode, 
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            timestep: int,
            spike_mode: str,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.T = timestep

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    timestep,
                    spike_mode,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # self.norm_f = nn.BatchNorm1d(d_model, **factory_kwargs) if not rms_norm else RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos  
        for layer in self.layers:         
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = residual
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

@MODELS.register_module()
class SpikePointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(SpikePointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.spike_mode = config.spike_mode
        self.T = config.timestep
        self.expand = config.expand
        self.label_smoothing = config.label_smoothing

        self.group_size = config.group_size
        self.num_group = config.num_group   
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size, expand=self.expand, timestep=self.T)

        self.encoder = Encoder(encoder_channel=self.encoder_dims, timestep=self.T, spike_mode=self.spike_mode)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(128, self.trans_dim, 1),
            nn.BatchNorm1d(self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 timestep=self.T,
                                 spike_mode=self.spike_mode,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(self.trans_dim * self.HEAD_CHANEL, 256, 1),
            nn.BatchNorm1d(256),
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(128, self.cls_dim, 1)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    def forward(self, pts):
        assert len(pts.shape) < 4, "shape of inputs is invalid"
        if self.spike_mode is not None:
            pts = (pts.unsqueeze(0)).repeat(self.T, 1, 1, 1) 

        # KNN & PointNet
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # TB G C 
        pos = self.pos_embed(center.flatten(0,1).permute(0,2,1)).permute(0,2,1).contiguous() # TB G C

        # Transformer
        x = group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, pos)

        # Head
        concat_f = x.mean(1).unsqueeze(-1)
        ret = self.cls_head_finetune(concat_f)
        ret = ret.unflatten(0,(self.T, -1)).mean(0)
        return ret.squeeze(2)


class SpikeMaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.spike_mode = config.spike_mode
        self.T = config.timestep
        self.expand = config.expand
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, timestep=self.T, spike_mode=self.spike_mode)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(128, self.trans_dim, 1),
            nn.BatchNorm1d(self.trans_dim),
        )
        
        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 timestep=self.T,
                                 spike_mode=self.spike_mode,
                                 rms_norm=self.config.rms_norm)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : T B G 3
            --------------
            mask : T B G (bool)
        '''
        T, B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.repeat(T, 1, 1).to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        # mask vis token 
        _, batch_size, seq_len, group_size, _ = neighborhood.size()
        x_vis = neighborhood[~bool_masked_pos].reshape(self.T, batch_size, -1, group_size, 3)
        x_vis = self.encoder(x_vis)   # B G C 

        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(-1, seq_len - int(self.mask_ratio * seq_len), 3)
        pos = self.pos_embed(masked_center.permute(0,2,1)).permute(0,2,1).contiguous() 

        # transformer
        x_vis = self.blocks(x_vis, pos)
        return x_vis, bool_masked_pos


class SpikeMambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.T = config.timestep
        self.spike_mode = config.spike_mode
        
        self.blocks = MixerModel(d_model=embed_dim,
                                n_layer=depth,
                                timestep=config.timestep,
                                spike_mode=None,   # ANN Decoder
                                rms_norm=config.rms_norm,
                                drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


@MODELS.register_module()
class SpikePoint_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.T = config.timestep
        self.spike_mode = config.spike_mode
        self.expand = config.expand
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = SpikeMaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.MAE_decoder = SpikeMambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size, expand=self.expand, timestep=self.T)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, **kwargs):
        # Convert to Spike
        assert len(pts.shape) < 4, "shape of inputs is invalid"
        if self.spike_mode is not None:
            pts = (pts.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
        
        neighborhood, center = self.group_divider(pts) 

        x_vis, mask = self.MAE_encoder(neighborhood, center)  # SNN Encoder
        TB, _, C = x_vis.shape  # TB VIS C
        
        # ANN Decoder
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(TB, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(TB, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(TB, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        TB, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(TB * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(TB * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        return loss1
