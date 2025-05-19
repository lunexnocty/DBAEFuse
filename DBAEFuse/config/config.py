from dataclasses import dataclass
from importlib import import_module
from os import path
from typing import Dict, Iterator, Optional, Tuple

import torch
import yaml
from dataset import VIFDataset, read_and_split
from torch.nn import Module, Parameter
from torch.utils.data import ConcatDataset, DataLoader


class ModelConfig:
    def __init__(self, target: str, params: dict, pretrained: Optional[str] = None) -> None:
        self.target: str = target
        self.params = {}
        for k, v in params.items():
            if self.check(v):
                self.params[k] = ModelConfig(**v)
            else:
                self.params[k] = v
        self.pretrained: Optional[str] = pretrained

    @staticmethod
    def check(param: dict) -> bool:
        if type(param) is dict:
            if {"target", "params"}.issubset(param.keys()):
                return True
        return False

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(target={self.target}, params={self.params}, pretrained={self.pretrained})"


class TrainArgs:
    def __init__(
        self,
        epochs: int,
        accumulation_steps: int,
        optimizer: dict,
    ) -> None:
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.optimizer = ModelConfig(**optimizer)

    def __repr__(self) -> str:
        return "\n".join(
            [
                "TrainArgs(",
                f"epochs={self.epochs}",
                f"accumulation_steps={self.accumulation_steps}",
                f"optimizer={self.optimizer}",
            ]
        )


@dataclass
class DatasetArgs:
    url: Dict[str, str]
    batch_size: int
    shuffle: bool
    ratio: Optional[Tuple[float, float] | Tuple[float, float, float]] = None


@dataclass
class TrainConfig:
    model: ModelConfig
    train: TrainArgs
    dataset: DatasetArgs


def load_config(file: str = "config/model.yaml") -> TrainConfig:
    assert path.isfile(file), FileNotFoundError
    with open(file) as f:
        config = yaml.full_load(f)
        model = ModelConfig(**config["model"])
        train = TrainArgs(**config["train"])
        dataset = DatasetArgs(**config["dataset"])
        return TrainConfig(model=model, train=train, dataset=dataset)


def load_model(config: ModelConfig, device: torch.device):
    kwargs = {}
    module_name, cls_name = config.target.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, cls_name)
    for k, v in config.params.items():
        kwargs[k] = load_model(v, device) if type(v) is ModelConfig else v
    model: Module = cls(**kwargs)
    if config.pretrained is not None and path.isfile(config.pretrained) and hasattr(model, "load_state_dict"):
        print(f"load from {config.pretrained}")
        state_dict = torch.load(config.pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def load_dataset(config: DatasetArgs):
    msrs_labels = read_and_split(config.url["MSRS"], config.ratio)
    tno_labels = read_and_split(config.url["TNO"], config.ratio)
    msrs_train = VIFDataset(msrs_labels.train)
    tno_train = VIFDataset(tno_labels.train)
    trainset = DataLoader(ConcatDataset([msrs_train, tno_train]), batch_size=config.batch_size, shuffle=config.shuffle)

    if msrs_labels.valid is None or tno_labels.valid is None:
        return [trainset]

    msrs_valid = VIFDataset(msrs_labels.valid)
    tno_valid = VIFDataset(tno_labels.valid)
    validset = DataLoader(ConcatDataset([msrs_valid, tno_valid]), batch_size=config.batch_size, shuffle=config.shuffle)
    if msrs_labels.test is None or tno_labels.test is None:
        return [trainset, validset]
    msrs_test = VIFDataset(msrs_labels.test)
    tno_test = VIFDataset(tno_labels.test)
    testset = DataLoader(ConcatDataset([msrs_test, tno_test]), batch_size=config.batch_size, shuffle=config.shuffle)
    return [trainset, validset, testset]


def load_optimizer(config: ModelConfig, params: Iterator[Parameter]):
    module_name, cls_name = config.target.rsplit(".", 1)
    module = import_module(module_name)
    optim = getattr(module, cls_name)
    return optim(params, **config.params)
