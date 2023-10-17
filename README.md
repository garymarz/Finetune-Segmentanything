# Finetune-Segmentanything
This is a demo fine-tune Segmentanything with Multiple GPUs. This case was trained on three GPUs and use `nn.DataParallel`
# Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.    
Install [Pytorch](https://pytorch.org/):    
Install Segment Anything:    
```
pip install git+https://github.com/facebookresearch/segment-anything.git    
cd Finetune-Segmentanything
pip install -e . 
```

# Checkpoint
Download the `ViT-L SAM` [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) 
# Data
trainingdata     
# Training    
`python train.py --batch 32 --dataroot ./trainingdata `
