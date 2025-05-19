import torch
from torch import Tensor, nn

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x_avg = self.fc(self.avg_pool(x))
        x_max = self.fc(self.max_pool(x))
        return self.sigmod(x_avg + x_max) * x

class SpatialAttentionLayer(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False, padding_mode='replicate')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True).values
        weights = torch.cat([x_avg, x_max], dim=1)
        return self.sigmoid(self.conv(weights)) * x

class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        self.channel_attn = ChannelAttentionLayer(channels, ratio)
        self.spatial_attn = SpatialAttentionLayer()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attn(x)
        return self.spatial_attn(x)
