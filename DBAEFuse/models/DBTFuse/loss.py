from typing import Literal

import torch
from models.filters import BilateralFilter, Gradient
from models.models import FuseInput, FuseOutput
from pytorch_msssim import MS_SSIM
from torch import Tensor, nn
from torch.nn import functional as F


class AELoss(nn.Module):
    def __init__(self, lamda: float = 100):
        super().__init__()
        self.lamda = lamda
        self.ms_ssim = MS_SSIM(data_range=1.0, channel=1)

    def pixel(self, x: Tensor, y: Tensor):
        return F.l1_loss(x, y)

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.ms_ssim(x, y)

    def forward(self, inputs: Tensor, outputs: Tensor):
        l_pixel = self.pixel(inputs, outputs)
        l_ssim = self.ssim(inputs, outputs)
        return l_pixel + self.lamda * l_ssim


class DBECFLoss(nn.Module):
    def __init__(self, alpha: float = 1, mu: float = 10):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.lamda = 100
        self.betas = (1.2, 4.5)

        self.bila = BilateralFilter()
        self.grad = Gradient()
        self.ssim = MS_SSIM(data_range=1.0, channel=1)
        self.aeloss = AELoss(self.lamda)

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
        return l_pixel + self.betas[0] * l_intensity + self.betas[1] * l_grad + self.lamda * self.ssim_loss(vi, ir, fu)

    def correlation_coefficient(self, x: Tensor, y: Tensor):
        centralized_x = x - x.mean()
        centralized_y = y - y.mean()
        nume = (centralized_x * centralized_y).sum()
        deno = torch.sqrt(centralized_x.square().sum() * centralized_y.square().sum())
        return nume / deno

    def feature_loss(self, c_v: Tensor, c_i: Tensor, p_v: Tensor, p_i: Tensor) -> Tensor:
        f_comm = (1 - self.correlation_coefficient(c_v, c_i)).abs()
        f_priv = self.correlation_coefficient(p_i, p_v).abs()
        return f_comm + f_priv

    def forward(self, inputs: FuseInput, outputs: FuseOutput) -> Tensor:
        vi = inputs.vi / 2 + 0.5
        ir = inputs.ir / 2 + 0.5

        fu = outputs.fusion / 2 + 0.5
        r_vi = outputs.vi_reconstruction / 2 + 0.5
        r_ir = outputs.ir_reconstruction / 2 + 0.5

        loss_ae = self.aeloss(vi, r_vi) + self.aeloss(ir, r_ir)

        loss_feature = self.feature_loss(
            outputs.vi_feature_commom, outputs.ir_feature_common, outputs.vi_feature_private, outputs.ir_feature_private
        )

        loss_fuse = self.fu_loss(vi, ir, fu)

        return loss_ae + self.mu * loss_feature + self.alpha * loss_fuse
