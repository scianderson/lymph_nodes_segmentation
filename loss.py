import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, true):
        # pred shape = [batch size, 2, 128, 128, 128], true shape = [2, 128, 128, 128]
        # for binary segmentation
        # eps = 1e-8
        # # if not squeeze, pred[:,0] is [batch,h,w,d], while true is [batch,1,h,w,d]
        # # pred_binary = torch.where(pred[:,0]>=0.5,1,0)
        # intersect_fg = pred[:,0] * true.squeeze()
        # union_fg = pred[:,0] + true.squeeze()
        # dice_val_fg = 2 * torch.sum(intersect_fg) / (torch.sum(union_fg)+eps)
        # dice_loss_fg = 1 - dice_val_fg

        # # intersect_bg = pred[:,1] * (1-true).squeeze()
        # # union_bg = pred[:,1] + (1-true).squeeze()
        # # dice_val_bg = 2 * torch.sum(intersect_bg) / (torch.sum(union_bg)+eps)
        # # dice_loss_bg = 1 - dice_val_bg

        # return dice_loss_fg#+dice_loss_bg

        return torch.norm(pred[:,0]-true.squeeze())+torch.norm(pred[:,1]-(1-true.squeeze()))