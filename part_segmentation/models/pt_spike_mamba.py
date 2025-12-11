from typing import Optional, Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation

from spikingjelly.clock_driven.neuron import *

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def build_spike_node(timestep, spike_mode, tau=2.0, v_threshold=0.5):
    if spike_mode == "lif":
        proj_lif = MultiStepLIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "elif":
        proj_lif = MultiStepEIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "plif":
        proj_lif = MultiStepParametricLIFNode(timestep=timestep, v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "if":
        proj_lif = MultiStepIFNode(timestep=timestep, v_threshold=v_threshold, detach_reset=True, backend="cupy")
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")
    return proj_lif


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
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, timestep, spike_mode, norm_cls=nn.BatchNorm1d, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.T = timestep
        self.mixer = mixer_cls(dim)
        # SNN: LIF / ANN:LN
        self.norm_lif = build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.LayerNorm(dim)
        
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
        hidden_states = self.norm_lif(residual)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


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


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, expand, timestep):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.T = timestep
        self.expand = expand
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: T B N 3
            ---------------------------
            output: T B G M 3
            center : T B G 3
        '''
        _, batch_size, num_points, _ = xyz.shape
        step_size_f = int((self.expand - 1.0)*self.num_group/self.T*2)
        step_size_b = int((self.expand - 1.0)*self.num_group)
        xyz = xyz.flatten(0,1)

        # fps the centers out
        # center = fps(xyz, self.num_group)  # B G 3                  # NOT Moving
        # center = fps(xyz, int(self.num_group*self.expand))  # B G' 3 # NOT Moving BUT Expanding
        # F:Whole Moving
        # center = fps(xyz[:batch_size], self.num_group + step_size_f*self.T) # B G' 3
        # center = torch.stack([center[:, (i * step_size_f):(i * step_size_f + self.num_group)] 
        #                              for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3
        # B:Sampling
        # center = fps(xyz[:batch_size], self.num_group + step_size_b*(self.T-1)) # B G' 3
        # center = torch.stack([torch.cat((center[:, :(self.num_group - step_size_b)], \
        #                     center[:, ((i-1) * step_size_b + self.num_group): \
        #                     (i * step_size_b + self.num_group)]), dim=1) \
        #                     for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3     
        # F:Whole Moving B:Sampling                                                                              
        center = fps(xyz[:batch_size], self.num_group + (step_size_f+step_size_b)*(self.T-1)) # B G' 3        
        center = torch.stack([torch.cat((center[:, i*step_size_f : i*step_size_f + (self.num_group - step_size_b)], \
                            center[:, ((i-1) * step_size_b + self.num_group + (self.T-1)*step_size_f):\
                            (i * step_size_b + self.num_group + (self.T-1)*step_size_f)]), dim=1) \
                            for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3         
        # F:Part Moving B:Sampling
        # center = fps(xyz[:batch_size], self.num_group + (step_size_f+step_size_b)*(self.T-1)) # B G' 3  
        # center = torch.stack([torch.cat((center[:, i*step_size_f:(i+1)*step_size_f], \
        #                     center[:, self.T*step_size_f : (self.num_group - step_size_b + (self.T-1)*step_size_f)], \
        #                     center[:, ((i-1) * step_size_b + self.num_group + (self.T-1)*step_size_f):\
        #                     (i * step_size_b + self.num_group + (self.T-1)*step_size_f)]), dim=1) \
        #                     for i in range(self.T)], dim=0).flatten(0,1)  # TB G 3  

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # TB G M
        idx_base = torch.arange(0, batch_size * self.T, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(self.T * batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(self.T, batch_size, -1, self.group_size, 3).contiguous()
        center = center.view(self.T, batch_size, -1, 1, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center
        return neighborhood, center.squeeze(-2)


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

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
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

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

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
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
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


class MixerModelForSegmentation(MixerModel):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
        self.residual_in_fp32 = residual_in_fp32

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

        self.fetch_idx = fetch_idx

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
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

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []

        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                residual_output = (hidden_states + residual) if residual is not None else hidden_states
                feature_list.append(residual_output)

        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim, config=None):
        super().__init__()

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = cls_dim

        self.spike_mode = config.spike_mode
        self.T = config.timestep
        self.expand = config.expand

        self.group_size = 32
        self.num_group = 256
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size, expand=self.expand, timestep=self.T)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims, timestep=self.T, spike_mode=self.spike_mode)
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            build_spike_node(self.T, self.spike_mode),
            nn.Conv1d(128, self.trans_dim, 1),
            nn.BatchNorm1d(self.trans_dim),
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)
        self.drop_path_rate = config.drop_path_rate
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False), # label can be seen as spike
                                        nn.BatchNorm1d(64)) # cancel LeakyRelu

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024], timestep=self.T, spike_mode=self.spike_mode)

        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.convs1_lif = build_spike_node(self.T, self.spike_mode)
        self.convs2_lif = build_spike_node(self.T, self.spike_mode)
        self.convs3_lif = build_spike_node(self.T, self.spike_mode)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            # import pdb; pdb.set_trace()
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

    def forward(self, pts, cls_label):
        B, N, C = pts.shape
        # Convert to Spike
        assert len(pts.shape) < 4, "shape of inputs is invalid"
        if self.spike_mode is not None:
            pts = (pts.unsqueeze(0)).repeat(self.T, 1, 1, 1) 

        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(center.flatten(0,1).permute(0,2,1)).permute(0,2,1).contiguous()

        # Final input
        x = group_input_tokens
        feature_list = self.blocks(x, pos)

        # Post-Fusion
        feature_list = [x.transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(self.T, B, -1).unsqueeze(-1).repeat(1, 1, 1, N)
        x_avg_feature = x_avg.view(self.T, B, -1).unsqueeze(-1).repeat(1, 1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(self.T, 1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 2)

        # Upsample & SegHead
        # import pdb; pdb.set_trace()
        f_level_0 = self.propagation_0(pts.flatten(0,1), center.flatten(0,1), pts.flatten(0,1), x.transpose(-2,-1))
        x = torch.cat((f_level_0, x_global_feature.flatten(0,1)), 1)
        x = self.bns1(self.convs1(self.convs1_lif(x)))
        x = self.dp1(x)  ### Dropout is necessary?
        x = self.bns2(self.convs2(self.convs2_lif(x)))
        x = self.convs3(self.convs3_lif(x))
        x = x.unflatten(0,(self.T, -1)).mean(0) # Time Avg
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
