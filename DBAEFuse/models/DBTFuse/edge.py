import torch
from models.base import DWConv2d
from torch import Tensor, nn


class Sobel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ]
        )
        self.sobel_x = DWConv2d(channels, sobel_x, padding=1, padding_mode="replicate")
        self.sobel_y = DWConv2d(channels, sobel_x.T, padding=1, padding_mode="replicate")

    def forward(self, x: Tensor) -> Tensor:
        return self.sobel_x(x) + self.sobel_y(x)


class Laplacian(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.dw = DWConv2d(channels, laplacian, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dw(x)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels

        self.basic = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="replicate")
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
        )
        self.sobel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding_mode="replicate"), Sobel(out_channels)
        )
        self.lap = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding_mode="replicate"), Laplacian(out_channels)
        )

        self.act = nn.LeakyReLU()

        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.basic(x) + self.expand(x) + self.sobel(x) + self.lap(x)
        out = self.act(self.norm(out))
        if self.use_residual:
            out = out + x
        return out
