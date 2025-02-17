import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SingleCenterLoss(nn.Module):
    def __init__(self, m = 0.3, D = 256, use_gpu=True):
        super(SingleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu
        self.l2loss = nn.MSELoss(reduce=False, reduction = 'none')

        if self.use_gpu:
            self.C = nn.Parameter(torch.randn(self.D).cuda())
        else:
            self.C = nn.Parameter(torch.randn(self.D))

    def forward(self, x, labels):
        batch_size = x.size(0)
        eud_mato = (self.l2loss(x, self.C.expand(batch_size, self.C.size(0))))
        eud_mat = torch.sqrt(eud_mato.sum(dim=1, keepdim=True))
        labels = labels.unsqueeze(1)
        real_count = labels.sum()
        dist_real = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()
        dist_fake = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()
        if real_count != 0:
            dist_real /= real_count
        if real_count != batch_size:
            dist_fake /= (batch_size - real_count)
        max_margin = dist_real - dist_fake + self.margin
        if max_margin < 0:
            max_margin = 0
        loss = dist_real + max_margin
        return loss, torch.abs(eud_mato)

'''
sgc_loss = SingleCenterLoss()
pred_feat = torch.rand((8,256), requires_grad=True).cuda()/256.0 # 实际代码不用/256.0，此处仅为演示过程防止数值溢出
label = torch.randint(0,2,size=(8,)).cuda()
loss = (sgc_loss(pred_feat, label)/16.0) # 实际使用有必要加除数限制loss大小
print('single center loss:',loss) 
'''
