from typing import Iterator, Sequence, TypeVar

import torch
from dataset import FuseInput, FuseOutput
from models.attention import CBAM
from models.edge import EdgeConvBlock
from torch import Tensor, nn

T = TypeVar("T", bound=Sequence)


def slice_window(it: T, size: int) -> Iterator[T]:
    for i in range(len(it) - size + 1):
        yield it[i : i + size]


class DBECFuse(nn.Module):
    def __init__(self, in_channels: int, channel_mults: list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.channel_mults = channel_mults
        self.num_features = self.in_channels * channel_mults[-1]

        self.encode_vi = self.get_encoder()
        self.encode_ir = self.get_encoder()
        self.encode_comm = self.get_encoder()
        self.attn_priv_vi = CBAM(self.num_features, ratio=4)
        self.attn_priv_ir = CBAM(self.num_features, ratio=4)
        self.decode = self.get_decoder()

    def get_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.channel_mults[0], 1),
            nn.LeakyReLU(),
            *[
                EdgeConvBlock(self.in_channels * cur, self.in_channels * nxt)
                for cur, nxt in slice_window(self.channel_mults, 2)
            ],
        )

    def get_decoder(self):
        return nn.Sequential(
            EdgeConvBlock(self.num_features * 2, self.num_features),
            *[
                EdgeConvBlock(self.in_channels * cur, self.in_channels * nxt)
                for cur, nxt in slice_window(list(reversed(self.channel_mults)), 2)
            ],
            nn.Conv2d(self.in_channels * self.channel_mults[0], self.in_channels, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: FuseInput):
        p_vi = self.encode_vi(inputs.vi)
        p_ir = self.encode_ir(inputs.ir)

        f_vi = self.encode_comm(inputs.vi)
        f_ir = self.encode_comm(inputs.ir)

        priv = self.attn_priv_vi(p_vi) + self.attn_priv_ir(p_ir)
        comm = torch.max(f_vi, f_ir)

        r_vi: Tensor = self.decode(torch.cat([p_vi, comm], dim=1))
        r_ir: Tensor = self.decode(torch.cat([p_ir, comm], dim=1))
        fuse: Tensor = self.decode(torch.cat([priv, comm], dim=1))

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
    net = DBECFuse(1, [1, 2, 4, 8])
    x = y = torch.randn(4, 1, 1024, 1024)
    print("start...")
    z = net(x, y)
    print(z.shape)
