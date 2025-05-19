import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

from typing import NamedTuple

import torch
from torch import Size, Tensor, nn


class PatchOutput(NamedTuple):
    patches: Tensor
    size: Size


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int | None = None) -> None:
        super().__init__()
        embed_dim = embed_dim or patch_size * patch_size * in_channels
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: Tensor) -> PatchOutput:
        patches: Tensor = self.patch(x)
        return PatchOutput(patches=patches.flatten(-2).transpose(-1, -2), size=x.shape[-2:])


class PatchUnEmbedding(nn.Module):
    def __init__(self, out_channels: int, patch_size: int, embed_dim: int | None = None) -> None:
        super().__init__()
        self.patch_size = patch_size
        embed_dim = embed_dim or patch_size * patch_size * out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.unpatch = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: Tensor, size: Size) -> Tensor:
        H, W = size
        x = x.transpose(-1, -2).view(x.size(0), -1, H // self.patch_size, W // self.patch_size)
        return self.unpatch(x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length=8092, learnable=False) -> None:
        super().__init__()
        freqs = 1.0 / (10000 ** (2.0 ** (torch.arange(0, embed_dim, 2) / embed_dim)))

        self.positional_embedding = nn.Parameter(torch.ones(max_seq_length, embed_dim), requires_grad=learnable)
        self.positional_embedding[:, 0::2] = freqs.sin()
        self.positional_embedding[:, 1::2] = freqs.cos()

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.positional_embedding[:seq_len].unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        patch_size: int = 16,
        num_heads: int = 32,
        num_layers: int = 2,
        embed_dim: int | None = None,
        learnable=False,
        dropout=0.0,
        bias=False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        embed_dim = embed_dim or patch_size * patch_size * in_channels
        dim_feedforward = embed_dim << 2
        encode_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            bias=bias,
        )
        layer_norm = nn.LayerNorm(embed_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encode_layer,
            num_layers=num_layers,
            norm=layer_norm,
            enable_nested_tensor=False,
            mask_check=False,
        )
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.patch_unembed = PatchUnEmbedding(out_channels, patch_size, embed_dim)
        self.positional_embed = SinusoidalPositionalEmbedding(embed_dim, learnable=learnable)

    def forward(self, x: Tensor):
        patched: PatchOutput = self.patch_embed(x)
        patches = patched.patches
        size = patched.size
        patches: Tensor = self.positional_embed(patches)
        patches = self.transformer(patches)
        return self.patch_unembed(patches, size)


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 1, 256, 256).cuda()
    net = Transformer(
        in_channels=1,
        out_channels=4,
        patch_size=16,
        num_heads=32,
        num_layers=4,
        embed_dim=1024,
        learnable=False,
        dropout=0.1,
        bias=False,
    ).cuda()
    y = net(x)
    print(x.shape, y.shape)
