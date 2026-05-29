# DisCo DETR

**Distance-aware Multi-view Contrastive Learning for DETR Pre-training**

Chao Ouyang, Yuyang Bai, Jun Zhang\*, Tianlu Gao, Jun Hao, Lijun Kong, David Wenzhong Gao.
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.*

📄 **Paper:** https://ojs.aaai.org/index.php/AAAI/article/view/39650

DisCo DETR is a self-supervised pre-training method for DETR-family detectors. Rather than bolting on external machinery, it **reuses DETR's own bipartite matching** as the supervision signal for contrastive learning, improving both localization and semantic representations at negligible cost.

## Highlights

- **Exploits DETR's native components** — no extra matching networks, no architectural changes; only ~0.16% additional per-epoch pre-training overhead.
- **State-of-the-art transfer among DETR pre-training methods** on COCO and PASCAL VOC across **five DETR variants** (Conditional DETR, Deformable DETR, DAB-DETR, RT-DETR, DINO), up to **+6.2 AP** over DETReg on RT-DETR at short schedules.

## Method

DisCo DETR pre-trains DETR-family detectors with two complementary, architecture-native objectives:

- **DMOQF — Distance-aware Multi-view Object Query Fusion** (localization): injects object features into object queries via distance-aware proposal matching across two augmented views, giving stable localization supervision in place of the random query injection used by prior work.
- **CLD — Contrastive Learning on Decoder outputs** (semantics): because DETR already performs bipartite matching between queries and objects, applying it independently to two views naturally yields positive query–output pairs for the same object — contrastive pairs at no extra matching cost.

Both objectives reuse components already present in DETR, so DisCo DETR adds no new matching networks and requires no architectural changes, and integrates into five DETR variants seamlessly.

## Code Structure

| Component | Path |
|---|---|
| DisCo DETR / RT-DETR algorithm | `openselfsup/models/DisCo_DETR.py`, `openselfsup/models/DisCo_RT_DETR.py` |
| DMOQF (multi-view query fusion) | `openselfsup/models/necks/{conditional_tr,deformable_tr,rt_tr}.py` |
| CLD (contrastive head) | `openselfsup/models/heads/{discodetr_head,discodetr_rt_head}.py` |
| Pre-training configs | `configs/selfsup/DisCo_DETR/main/`, `.../ablations/` |
| Training entry point | `tools/train.py` |
| Downstream fine-tuning (Conditional DETR) | `downstream_finetune/conditionaldetr/` |

## Installation

Tested with Python 3.8, PyTorch 1.8.1, CUDA 11.7, cuDNN 7.6.5 (Conditional / Deformable DETR pre-training). The codebase is built on [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup) and uses `mmcv`.

```bash
conda create -n discodetr python=3.8 -y
conda activate discodetr

# install a torch build matching your CUDA first, e.g.:
pip install torch==1.8.1 torchvision==0.9.1

pip install -r requirements.txt
python setup.py develop
```

- **Deformable attention:** compile the CUDA operator following the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) repo.
- **RT-DETR variant:** requires `torch>=2.0`; see the RT-DETR section in `requirements.txt`.

## Data Preparation

Pre-training uses ImageNet with EdgeBox proposals; downstream evaluation uses COCO / PASCAL VOC. Place datasets under `data/` (git-ignored), e.g.:

```
data/
├── imagenet/
├── coco/
└── VOCdevkit/
```

Exact paths are defined in the config files under `configs/selfsup/DisCo_DETR/`.

## Pre-training

```bash
python tools/train.py \
  configs/selfsup/DisCo_DETR/main/r50_condtr_300q_syncbn_imgnet_edgebox_rdiscr_gdiscr_gpu32_step30w.py \
  --work_dir work_dirs/disco_condtr_300q
```

`main/` holds the main experiments and `ablations/` the ablation configs. Multi-GPU / multi-node training follows the standard OpenSelfSup / mmcv launchers.

## Downstream Fine-tuning

An example of fine-tuning a pre-trained backbone with Conditional DETR is provided in `downstream_finetune/conditionaldetr/`. Other DETR variants follow their official fine-tuning pipelines.

## Citation

```bibtex
@inproceedings{ouyang2026discodetr,
  title     = {DisCo DETR: Distance-aware Multi-view Contrastive Learning for DETR Pre-training},
  author    = {Ouyang, Chao and Bai, Yuyang and Zhang, Jun and Gao, Tianlu and Hao, Jun and Kong, Lijun and Gao, David Wenzhong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

## Acknowledgement

Built on [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup). The DETR variants build on [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), and [DINO](https://github.com/IDEA-Research/DINO). We thank the authors for releasing their code.

## License

This project inherits the Apache-2.0 License from OpenSelfSup. Please also respect the upstream licenses of the respective DETR codebases.
