import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss functions
class EASTLoss(nn.Module):
    def __init__(self):
        super(EASTLoss, self).__init__()
        
    def forward(self, score_map_pred, geo_map_pred, score_map_gt, geo_map_gt, training_mask):
        # Apply training mask to both predictions and ground truth
        training_mask = training_mask.float()
        
        # Calculate score map loss (dice loss)
        eps = 1e-5
        score_map_pred = score_map_pred * training_mask
        score_map_gt = score_map_gt * training_mask
        
        intersection = torch.sum(score_map_pred * score_map_gt) # phần chung của cả 2
        union = torch.sum(score_map_pred) + torch.sum(score_map_gt) + eps # phần cả 2 đều có
        dice_loss = 1 - 2 * intersection / union
        

        # QUAD loss (smooth L1 for 8 coordinates)
        geo_map_pred = geo_map_pred * training_mask.unsqueeze(1)
        geo_map_gt = geo_map_gt * training_mask.unsqueeze(1)
        
        geo_loss = F.smooth_l1_loss(geo_map_pred, geo_map_gt, reduction='none')
        geo_loss = torch.sum(geo_loss * training_mask.unsqueeze(1)) / (torch.sum(training_mask) + eps)
        
        # Total loss (weighted sum)
        loss = dice_loss + 10 * geo_loss
        
        return loss, dice_loss, geo_loss
    
# Loss functions
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, score_map_pred, score_map_gt):
        # Calculate score map loss (dice loss)
        eps = 1e-5
        
        intersection = torch.sum(score_map_pred * score_map_gt, dim=1) # phần chung của cả 2
        union = torch.sum(score_map_pred,dim=1) + torch.sum(score_map_gt,dim=1)  # phần cả 2 đều có
        dice_loss = 1 - (2 * intersection  + eps) / (union + eps)
        
        return dice_loss.mean()