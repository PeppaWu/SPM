# Training

## Pre-train

```shell
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/spike_pretrain.yaml --exp_name --exp_name <name>
```
## Classification on ModelNet40

```shell
CUDA_VISIBLE_DEVICES=<GPU> --scratch_model --config cfgs/spike_finetune_modelnet.yaml --exp_name <name>
```
## Classification on ScanObjectNN

```shell
# 0. ScanObjectNN OBJ-BG
CUDA_VISIBLE_DEVICES=<GPU> python main.py --scratch_model --config cfgs/spike_finetune_scan_ogjbg.yaml --exp_name <name>

# 1. ScanObjectNN OBJ-ONLY
CUDA_VISIBLE_DEVICES=<GPU> python main.py --scratch_model --config cfgs/spike_finetune_scan_objonly.yaml --exp_name <name>

# 2. ScanObjectNN PB-T50-RS
CUDA_VISIBLE_DEVICES=<GPU> python main.py --scratch_model --config cfgs/spike_finetune_scan_hardest.yaml --exp_name <name>
```