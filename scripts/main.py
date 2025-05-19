import os, time, argparse
from typing import Callable, Iterator

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from PIL import Image

from config import TrainConfig, load_config, load_dataset, load_model, load_optimizer
from dataset import FuseInput, FuseOutput, parse_directory, Tensor_to_PIL, PIL_to_Tensor
from models.models import DBECFuse
from models.loss import DBECFLoss

def parse_commandline():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='train model')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='test model')
    parser.add_argument('--save', '-s', dest='save', action='store', default='0.1', help='model save path')
    return parser.parse_args()


def timer(fn: Callable):
    def warpper(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        used_time = round(time.time() - start_time)
        minutes, seconds = divmod(used_time, 60)
        print(f'{fn.__name__} {minutes}:{seconds}s', end=' ')
        return ret
    return warpper

def train(
        config: TrainConfig,
        device: torch.device,
        save_path: str
    ):
    model: DBECFuse = load_model(config.model, device)
    
    criterion = DBECFLoss(alpha=0.5)
    criterion.to(device)
    
    trainloader, validloader = load_dataset(config.dataset)
    optim = load_optimizer(config.train.optimizer, model.parameters())
    accumulation_steps = config.train.accumulation_steps
    best_valid_loss = float('inf')
    print(f'train <{model._get_name()}> on {device.type}')
    for epoch in range(config.train.epochs):
        train_loss = train_loop(
            model, criterion, trainloader,
            optim, accumulation_steps, device
        )
        valid_loss = valid_loop(
            model, criterion, validloader,
            device
        )
        print(f'Epoch {epoch + 1}: train loss {train_loss}, valid loss {valid_loss}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint(model, save_path)
        torch.cuda.empty_cache()

def test(config: TrainConfig, path: str, device: torch.device):
    model: DBECFuse = load_model(config.model, device)
    
    model_name = config.model.pretrained.split('/', 1)[-1].replace('/', '_').removesuffix('.pth')
    save_path = os.path.join(path, f'{model_name}')
    os.makedirs(save_path, exist_ok=True)
    
    model = model.eval()
    with torch.no_grad():
        for (vi, ir, filename) in load_data(path):
            inputs = FuseInput(vi.to(device), ir.to(device))
            output: FuseOutput = model(inputs)

            image: Image.Image = Tensor_to_PIL(output.fusion.squeeze(0))

            save_file = os.path.join(save_path, filename)
            image.save(save_file)
            print(f'saved: {save_file}', end='\r')

def load_data(dir: str) -> Iterator[tuple[Tensor, Tensor, str]]:
    labels = parse_directory(dir)
    for item in labels:
        vi = PIL_to_Tensor(Image.open(item['vi']).convert('L')).unsqueeze(0) # type: ignore
        ir = PIL_to_Tensor(Image.open(item['ir']).convert('L')).unsqueeze(0) # type: ignore
        yield vi, ir, item['vi'].rsplit('\\', 1).pop()

@timer
def train_loop(
    model: DBECFuse,
    criterion: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    accumulation_steps: int,
    device: torch.device,
) -> float:
    train_loss = 0.0
    model.train()
    for step, (vi, ir) in enumerate(trainloader):
        inputs = FuseInput(vi.to(device), ir.to(device))
        outputs: FuseOutput = model(inputs)
        loss: Tensor = criterion(inputs, outputs)
        train_loss += loss.item()
        loss = loss / accumulation_steps
        loss.backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
    return train_loss / len(trainloader)

@timer
def valid_loop(
    model: DBECFuse,
    criterion: nn.Module,
    validloader: DataLoader, 
    device: torch.device
) -> float:
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, (vi, ir) in enumerate(validloader):
            inputs = FuseInput(vi.to(device), ir.to(device))
            outputs: FuseOutput = model(inputs)
            loss: Tensor = criterion(inputs, outputs)
            valid_loss += loss.item()
        return valid_loss / len(validloader)

def checkpoint(model: nn.Module, version: str):
    model_name = model.__class__.__name__
    filepath = f'weights/{model_name}/v{version}.pth'
    if os.path.exists(filepath): os.remove(filepath)
    torch.save(model.state_dict(), filepath)


if __name__ == '__main__':
    config_path = 'config/model.yaml'
    config = load_config(config_path)

    args = parse_commandline()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.test:
        test(config, 'E:/projects/yolov8/datasets/RoadScene_Yolo/images', device)
    elif args.train:
        train(config, device, args.save)