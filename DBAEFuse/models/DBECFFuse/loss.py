from typing import Literal
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from pytorch_msssim import MS_SSIM
from models.filters import BilateralFilter, Gradient
from dataset import FuseInput, FuseOutput

class AELoss(nn.Module):
    def __init__(self, betas: tuple[float, float] = (1, 100)):
        super().__init__()
        self.betas = betas
        self.ms_ssim = MS_SSIM(data_range=1., channel=1)
    
    def pixel(self, x: Tensor, y: Tensor):
        return F.l1_loss(x, y)

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.ms_ssim(x, y)

    def forward(self, inputs: Tensor, outputs: Tensor):
        l_pixel = self.pixel(inputs, outputs)
        l_ssim = self.ssim(inputs, outputs)
        return self.betas[0] * l_pixel + self.betas[1] * l_ssim


class DBECFLoss(nn.Module):
    def __init__(self, alpha: float = 1):
        super().__init__()
        self.alpha = alpha
        self.betas = (1.2, 4.5, 100)

        self.bila = BilateralFilter()
        self.grad = Gradient()
        self.ssim = MS_SSIM(data_range=1.0, channel=1)
        self.aeloss = AELoss()

    def intensity_loss(self, vi: Tensor, ir: Tensor, fu: Tensor):
        return F.l1_loss(fu, torch.max(vi, ir))

    def grad_loss(self, vi: Tensor, ir: Tensor, fu: Tensor):
        return F.l1_loss(self.grad(fu), torch.max(self.grad(self.bila(ir)), self.grad(vi)))
    
    def ssim_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        return 1 - 0.5 * (self.ssim(fu, vi) + self.ssim(fu, ir))
    
    def fu_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        l_pixel = F.l1_loss(fu, (vi + ir * 2) / 3)
        l_intensity = self.intensity_loss(vi, ir, fu)
        l_grad = self.grad_loss(vi, ir, fu)
        l_ssim = self.ssim_loss(vi, ir, fu)
        return l_pixel \
             + self.betas[0] * l_intensity \
             + self.betas[0] * l_grad \
             + self.betas[1] * l_ssim
    
    def forward(self, inputs: FuseInput, outputs: FuseOutput) -> Tensor:
        vi = inputs.vi / 2 + 0.5
        ir = inputs.ir / 2 + 0.5

        fu = outputs.fusion / 2 + 0.5
        r_vi = outputs.vi_reconstruction / 2 + 0.5
        r_ir = outputs.ir_reconstruction / 2 + 0.5

        loss_ae = self.aeloss(vi, r_vi) \
                + self.aeloss(ir, r_ir) \
                + 10 * F.l1_loss(outputs.vi_feature_commom, outputs.ir_feature_common)

        loss_fuse = self.fu_loss(vi, ir, fu)

        return loss_ae + self.alpha * loss_fuse
