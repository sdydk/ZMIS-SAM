# ZMIS-SAM: Zooplankton Marine Instance Segmentation with SAM

ZMIS-SAM adapts the [Segment Anything Model (SAM)](https://segment-anything.com/) for fine-grained instance segmentation of marine zooplankton. It uses a frozen SAM ViT-Huge backbone with lightweight adapters and a Mask R-CNN–style detection head, enabling precise detection and segmentation of 47 zooplankton categories.

![Model Architecture](figs/ZMIS-SAM.png)


## :speech_balloon: Updates
🚩 **News** (2026.06) This paper has been accepted as a paper at [**_ECCV 2026_**]. The files corresponding to ZMIS-SAM are currently incomplete, and we are uploading relevant files step by step.

## Dataset: ZMIS5K

ZMIS5K contains 5,358 microscopic zooplankton images across 47 species, annotated in COCO format with bounding boxes and instance masks.

![Dataset Overview](figs/ZMIS5K.png)

<details>
<summary>47 Categories</summary>

Calanus sinicus, Sagitta crassa, Themisto gracilipes, Penilia avirostris, Centropages abdominalis, Acartia pacifica, Centropages tenuiremis, Pontellopsis tenuicauda, Calanopia thompsoni, Sugiura chengshanense, Ophioplutues larva early, Eirene menoni, Macrura larva, Evadne tergestina, Muggiaea atlantica, Paracalanus parvus, Oithona plumifera, Pleurobrachia globosa, Clytia folleata, Obelia dichotoma, Ectopleura bimanatus, Dolioletta gegenbauri, Oikopleura longicauda, Tornaria larva, Polychaeta larva early, Polychaeta larva later, Turritopsis nutricula, Proboscidactyla flavicirrata, Fritillaria formica, Labidocera rotunda, Alima larva, Megalopa larva, Brachyura zoea larva, Ophioplutues larva later, Fish eggs, Fish larva, Actinotrocha larva, Trochophora larva, Bougainvillia muscus, Aequorea conica, Varitentaculata yantaiensis, Porcellana zoea larva, Acetes larva, Centropages dorsispinatus, Clytia hemisphaerica, Jellyfish larva, Remains

</details>

## Model Architecture

- **Backbone**: SAM ViT-Huge (frozen decoder, pretrained)
- **Adapter**: ZViTAdapters inserted at ViT layers 8–32 (every 2 layers), embed_dim=1280
- **Neck**: ZMISFPN — ZMISFeatureAggregator (multi-layer feature fusion) + ZMISSimpleFPNHead (5-scale output)
- **Detection Head**: RPNHead + ZMISPrompterAnchorRoIPromptHead (Mask R-CNN style)
- **Mask Head**: ZMISPrompterAnchorMaskHead with ZMISSamMaskDecoder

## Installation

```bash
pip install -U openmim
mim install mmengine mmcv mmdet

git clone https://github.com/your-repo/ZMIS-SAM.git
cd ZMIS-SAM
pip install -e .
```

Download SAM ViT-Huge pretrained weights and place them at:
```
pretrain/sam-vit-huge/pytorch_model.bin
```

## Data Preparation

Organize the ZMIS5K dataset as follows:

```
data/
├── train/
├── val/
├── test/
└── annotations/
    ├── train_annotations.json
    ├── val_annotations.json
    └── test_annotations.json
```

Update `data_root` in [configs/zmis_train.py](configs/zmis_train.py#L42) to your local path.

## Training

```bash
# Single GPU
python tools/train.py configs/zmis_train.py

# Multi-GPU (e.g., 4 GPUs)
bash tools/dist_train.sh configs/zmis_train.py 4
```

Key training settings ([configs/zmis_train.py](configs/zmis_train.py)):

| Setting | Value |
|---|---|
| Input size | 512 × 512 |
| Epochs | 400 |
| Optimizer | AdamW (lr=2e-4, wd=0.05) |
| LR schedule | Linear warmup + Cosine annealing |
| Precision | AMP float16 |
| Batch size | 2 per GPU |

## Evaluation

```bash
# Single GPU
python tools/test.py configs/zmis_train.py <checkpoint> --eval bbox segm

# Multi-GPU
bash tools/dist_test.sh configs/zmis_train.py <checkpoint> 4
```

Metrics reported: COCO bbox mAP and segm mAP.

## Citation
If you find our repo or ZMIS5K dataset useful for your research, please cite us:
```
@inproceedings{dekunyuan,
  title     = {ZMIS-SAM: Segment Anything Model Enhanced with Wavelet Transform for Zooplankton Microscopy Image Instance Segmentation},
  author    = {Yuan, Dekun and Li, Zhongwei and Qiao, Zheng and Zhang, jie},
  booktitle = {Proceedings of the 19th European Conference on Computer Vision},
  pages     = {},
  year      = {2026}
  url       = {},
}
```

## Acknowledgements

This project builds on [MMDetection](https://github.com/open-mmlab/mmdetection) and [Segment Anything Model](https://github.com/facebookresearch/segment-anything). In addition, we referenced some of the code in the [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/lightning) and [USIS-SAM](https://github.com/LiamLian0727/USIS10K/tree/main) repository. Thanks to them for their excellent work.
