from dataclasses import dataclass
from itertools import pairwise

import torch
from models.attention import CBAM
from models.edge import EdgeConvBlock
from models.transformer import Transformer
from torch import Tensor, nn


@dataclass
class FuseInput:
    vi: Tensor
    ir: Tensor


@dataclass
class FuseOutput:
    fusion: Tensor
    vi_reconstruction: Tensor
    ir_reconstruction: Tensor
    vi_feature_commom: Tensor
    ir_feature_common: Tensor
    vi_feature_private: Tensor
    ir_feature_private: Tensor


class ECB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels_list: list[int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj_in = nn.Conv2d(in_channels, channels_list[0], 1)
        self.blocks = nn.Sequential(*[EdgeConvBlock(cur, nxt) for cur, nxt in pairwise(channels_list)])
        self.proj_out = nn.Conv2d(channels_list[-1], out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_out(self.blocks(self.proj_in(x)))


class DBTFuse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        channels_list: list[int],
        patch_size: int,
        num_heads: int,
        num_layers: int,
        embed_dim: int,
        dropout=0.0,
        bias=False,
    ):
        super().__init__()
        self.channels_list = channels_list
        self.in_channels = in_channels

        self.encode_vi = ECB(in_channels, num_features, channels_list)
        self.encode_ir = ECB(in_channels, num_features, channels_list)
        self.encode_comm = Transformer(
            in_channels=in_channels,
            out_channels=embed_dim // (patch_size * patch_size),
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            embed_dim=embed_dim,
            learnable=False,
            dropout=dropout,
            bias=bias,
        )
        self.decode = ECB(
            in_channels=num_features + self.encode_comm.out_channels,
            out_channels=out_channels,
            channels_list=channels_list[::-1],
        )

        self.attn_vi = CBAM(self.encode_ir.out_channels)
        self.attn_ir = CBAM(self.encode_ir.out_channels)
        self.proj_out = nn.Tanh()

    def forward(self, inputs: FuseInput):
        p_vi = self.encode_vi(inputs.vi)
        p_ir = self.encode_ir(inputs.ir)
        f_vi = self.encode_comm(inputs.vi)
        f_ir = self.encode_comm(inputs.ir)

        priv = self.attn_vi(p_vi) + self.attn_ir(p_ir)
        comm = torch.max(f_vi, f_ir)
        r_vi: Tensor = self.proj_out(self.decode(torch.cat([p_vi, comm], dim=1)))
        r_ir: Tensor = self.proj_out(self.decode(torch.cat([p_ir, comm], dim=1)))
        fuse: Tensor = self.proj_out(self.decode(torch.cat([priv, comm], dim=1)))

        return FuseOutput(
            fusion=fuse.clamp(-1.0, 1.0),
            vi_reconstruction=r_vi.clamp(-1.0, 1.0),
            ir_reconstruction=r_ir.clamp(-1.0, 1.0),
            vi_feature_commom=f_vi,
            ir_feature_common=f_ir,
            vi_feature_private=p_vi,
            ir_feature_private=p_ir,
        )

    def frozen(self, *__names: str):
        for name, param in self.named_parameters():
            for _name in __names:
                if _name in name:
                    param.requires_grad = False
                    print("frozen layer:", name)


if __name__ == "__main__":
    model = DBTFuse(
        in_channels=1,
        out_channels=1,
        num_features=64,
        channels_list=[4, 8, 16, 16],
        patch_size=16,
        num_heads=32,
        num_layers=4,
        embed_dim=1024,
        dropout=0.1,
        bias=False,
    )

    vi = torch.randn(1, 1, 256, 256)
    ir = torch.randn(1, 1, 256, 256)

    from thop import clever_format, profile

    FLOPs, Params = profile(model, inputs=FuseInput(vi, ir))
    print(FLOPs, Params)
    FLOPs, Params = clever_format([FLOPs * 2, Params], "%.4f")
    print(FLOPs, Params)
