import os
# set up visible device
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,5"

import numpy as np
import os
import cv2
join = os.path.join

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from collections.abc import Callable
from torch.nn.modules.loss import _Loss
from glob import glob
import json
import argparse
from skimage import transform, io
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(2023)
np.random.seed(2023)


class NpzDataset(Dataset): 
    def __init__(self, data_root, root):
        self.data_root = data_root
        self.npz_files = glob(os.path.join(data_root,root)+'\\*.json')
        self.d = {'defect':1,'Com':1, 'Date':1, 'Drain':1, 'Drainsub':1, 'Dsource':1, 'Gate':1, 'IN':1, 'MIT':1, 'Mashcom':1, 'Source':1}
        print('Loading',data_root)
        
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        img_embed = self.npz_files[index]
        d = open(img_embed,'r')
        data = json.load(d)
        image = cv2.imread(self.data_root+'\\images\\'+data["image_name"])
        image = cv2.resize(image,(1024,1024))
        gt2D = io.imread(self.data_root+'\\masks\\'+data["gt_data"])
        gt2D = transform.resize(
            gt2D == 255,
            (1024, 1024),
            order=0,
            preserve_range=True,
            mode="constant")
        gt2D = np.uint8(gt2D)
        H, W = gt2D.shape
        
        y_indices, x_indices = np.where(gt2D > 0)
        points = [[x,y] for y,x in zip(*np.where(gt2D > 0))]
        if len(points) > 0:
            points = points[int(len(points)/2)]
        else:
            points = [0,0]
        if x_indices.size==0:
            x_min, x_max = 0,0
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
        if y_indices.size==0:
            y_min, y_max = 0,0
        else:
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        x = np.random.randint(1, 10)
        y = np.random.randint(1, 10)
        x_min = max(0, x_min - x)
        x_max = min(W, x_max + x)
        y_min = max(0, y_min - y)
        y_max = min(H, y_max + y)
        box_np = np.array([x_min, y_min, x_max, y_max])
        points = np.array([points])
        
        sam_trans = ResizeLongestSide(1024)
        image_data_pre =  sam_trans.apply_image(image)
        image_data_pre = torch.as_tensor(image_data_pre.transpose(2, 0, 1))
        image_data_pre = sam_model.preprocess(image_data_pre)
        box = sam_trans.apply_boxes(box_np, (H, W))
        points = sam_trans.apply_coords(points, (H, W))

        return image_data_pre.float(), torch.Tensor(box).float(), torch.Tensor(points).float(), torch.Tensor([self.d[data["labels"]]]).float(), torch.Tensor([gt2D.tolist()]).long()

class FULLSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, pt):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = box
            
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=pt,
                boxes=box_torch,
                masks=None,)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,)
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,)
        
        return ori_res_masks
    
def dice_loss(inputs, labels, num_masks):
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss

def sigmoid_focal_loss(inputs, labels, num_masks, alpha = 0.25, gamma = 2):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs.sigmoid()
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="vit_l")
    parser.add_argument("--checkpoint", type=str, default="vit_l.pth")
    
    parser.add_argument("--model_save_path", type=str, default="fintune_sam")
    parser.add_argument("--data_root", type=str, default="D:\\yangu\\dataset\\SAM_trianingdata") # "D:\\sam\\dataset\\SAM_trianingdata"
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model_type = args.model_type
    checkpoint = args.checkpoint
    c = 0
    model_save_path = args.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    medsam_model = FULLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder)
    
    medsam_model = nn.DataParallel(medsam_model, device_ids=[0,1,2]).to(device)
    # medsam_model = nn.DataParallel(medsam_model).to(device)
    # Set up the optimizer, hyperparameter tuning will improve performance here
    # optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0.001)
    medsam_model.train()
    img_mask_encdec_params = list(medsam_model.module.image_encoder.parameters()) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay)
    # seg_loss = dice_loss()
    # ce_loss = sigmoid_focal_loss()
    num_epochs = 1000
    losses = []
    best_loss = 100
    
    train_dataset = NpzDataset(args.data_root, 'train')
    val_dataset = NpzDataset(args.data_root, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(num_epochs):
        epoch_loss = 0
        print('Training SAM epoch:',epoch)
        focal, dice, val_focal, val_dice, val_epoch_loss, mask_iou, seg = 0, 0, 0, 0, 0, 0, 0
        # train
        s = 0
        for steps, (image, box, points, labels, gt2D) in enumerate(train_dataloader):
            s += 1
            image = image.to(device)
            pt = (points.to(device), labels.to(device))
            num_masks = sum([len(b) for b in box])
            box = box.to(device)
            mask_predictions = medsam_model(image, box, pt)
            
            masks = mask_predictions.squeeze(1).flatten(1)
            label = gt2D.to(device).flatten(1)
            loss_focal = sigmoid_focal_loss(masks, label.float(), num_masks)
            loss_dice = dice_loss(masks, label.float(),num_masks)
            loss = loss_dice + loss_focal
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            focal += loss_focal.item()
            dice += loss_dice.item()
        epoch_loss /= s
        focal /= s
        dice  /= s
        print(f'EPOCH: {epoch}, Loss: {round(epoch_loss, 4)}, focal_loss: {round(focal, 4)}, dice_loss: {round(dice, 4)}')
        # if epoch % 1  == 0:
        #     val_focal, val_dice, val_epoch_loss, val_iou, b_loss = 0, 0, 0, 0, 0
        #     for step, (image, box, points, labels, gt2D) in enumerate(tqdm(val_dataloader)):
        #         image = image.to(device)
        #         pt = (points.to(device), labels.to(device))
        #         #image = sam_model.preprocess(image)
        #         points, labels = points.to(device), labels.to(device)
        #         mask_predictions = medsam_model(image, box, pt)
        #         num_masks = sum([len(b) for b in box])
        #         masks = mask_predictions.squeeze(1).flatten(1)
        #         label = gt2D.to(device).flatten(1)
        #         val_loss_focal = seg_loss(masks, label)
        #         val_loss_dice = ce_loss(masks, label.float())
        #         val_loss = val_loss_dice + val_loss_focal
        #         val_epoch_loss += val_loss.item()
        #         val_focal += val_loss_focal.item()
        #         val_dice += val_loss_dice.item()
        #     val_epoch_loss /= (step+1)
        #     val_focal /= (step+1)
        #     val_dice /= (step+1)
        #     print(f'EPOCH: {epoch}, VAL_Loss: {round(val_epoch_loss, 4)}, VAL_focal_loss: {round(val_focal, 4)}, VAL_dice_loss: {round(val_dice , 4)}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            c += 1
            print('best_loss', best_loss)
            torch.save(sam_model.state_dict(), join(model_save_path, 'sam_{}.pth'.format(c)))
