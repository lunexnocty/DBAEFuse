import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

PIL_to_Tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])

Tensor_to_PIL = transforms.Compose([transforms.Lambda(lambda x: x / 2 + 0.5), transforms.ToPILImage()])


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


random.seed("amane rinne")


class VIFDataset(Dataset):
    """红外-可见光融合数据集"""

    def __init__(self, labels: List[Dict[str, str]], *, vi_transform=None, ir_transform=None) -> None:
        super().__init__()
        self.labels = labels
        default_transform = transforms.Compose([transforms.CenterCrop(256), PIL_to_Tensor])
        self.vi_transform = vi_transform or PIL_to_Tensor
        self.ir_transform = ir_transform or PIL_to_Tensor

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.labels[index]
        vi = Image.open(item["vi"]).convert("L")
        ir = Image.open(item["ir"]).convert("L")
        with torch.no_grad():
            return self.vi_transform(vi), self.ir_transform(ir)

    def __len__(self):
        return len(self.labels)


@dataclass
class ImageLabels:
    train: List[Dict[str, str]]
    valid: Optional[List[Dict[str, str]]] = None
    test: Optional[List[Dict[str, str]]] = None


def read_and_split(root: str, ratios: Tuple[float, float] | Tuple[float, float, float] | None):
    labels = parse_directory(root)
    if ratios is None:
        return ImageLabels(labels)
    assert sum(ratios) == 1, "划分数据集的概率之和必须为 1"
    random.shuffle(labels)
    size = len(labels)
    train_offset = math.ceil(size * ratios[0])
    trainset = labels[:train_offset]
    if len(ratios) == 2:
        validset = labels[train_offset:]
        return ImageLabels(trainset, validset)
    else:
        valid_offset = math.floor(size * ratios[1]) + train_offset
        validset = labels[train_offset:valid_offset]
        testset = labels[valid_offset:]
        return ImageLabels(trainset, validset, testset)


def parse_directory(root: str) -> List[Dict[str, str]]:
    classes = os.listdir(root)
    assert "vi" in classes, "缺少可见光数据集"
    assert "ir" in classes, "缺少红外数据集"

    labels = []
    for filename in os.listdir(vipath := os.path.join(root, "vi")):
        if os.path.isfile(irfile := os.path.join(root, "ir", filename)):
            labels.append({"vi": os.path.join(vipath, filename), "ir": irfile})
    return labels


class Color:
    def __init__(self, kr=0.299, kb=0.114):
        self.kr = kr
        self.kb = kb

    def RGB_to_YCbCr(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r, g, b = image.chunk(3, dim=0)
        y = self.kr * r + (1 - self.kr - self.kb) * g + self.kb * b
        cb = 0.5 * (b - y) / (1 - self.kb)
        cr = 0.5 * (r - y) / (1 - self.kr)
        return y, cb, cr

    def YCbCr_to_RGB(self, y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
        r = y + 2 * cr * (1 - self.kr)
        b = y + 2 * cb * (1 - self.kb)
        g = (y - self.kr * r - self.kb * b) / (1 - self.kb - self.kr)
        return torch.cat([r, g, b], dim=0)
