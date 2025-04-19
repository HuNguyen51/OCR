import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx
        
    def forward(self, pred, target):
        # Tính log_softmax một lần
        log_probs = F.log_softmax(pred, dim=self.dim)
        
        # Tạo mask cho padding tokens
        padding_mask = (target != self.padding_idx).float().unsqueeze(1)
        
        # Tạo one-hot encoding
        target_one_hot = torch.zeros_like(log_probs).scatter_(
            1, target.unsqueeze(1), 1.0
        )
        
        # Áp dụng label smoothing
        smooth_target = target_one_hot * self.confidence + \
                        (1 - target_one_hot) * self.smoothing / (self.cls - 2)

        # Đặt giá trị của padding_idx thành 0
        smooth_target[:, self.padding_idx] = 0
        
        # Nhân với padding_mask để loại bỏ ảnh hưởng của padding tokens
        loss = -torch.sum(smooth_target * log_probs * padding_mask, dim=self.dim)
        
        # Chia cho số lượng token không phải padding trong mỗi mẫu
        non_padding_tokens = padding_mask.sum(dim=self.dim).clamp(min=1.0)
        loss = loss / non_padding_tokens
        
        return loss.mean()