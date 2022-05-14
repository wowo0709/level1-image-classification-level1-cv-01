import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

import time


def load_model(saved_model, model, num_classes, device):
    model_cls = getattr(import_module("model"), model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dirs, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    models = []
    for idx, model_dir in enumerate(model_dirs):
        models.append(load_model(model_dir, args.models[idx], num_classes, device).to(device))
        models[idx].eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = torch.Tensor([[0 for _ in range(num_classes)] for _ in range(len(images))]).to(device)
            for w, model in zip(args.ensemble_weights,models):
                pred += w * model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    nicknames = [name.split('/')[-2] for name in model_dirs]
    nickname = '_'.join(nicknames)
    time = time.time()
    info.to_csv(os.path.join(output_dir, f'ensemble_{nickname}_{time}_output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[300, 300], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--num_models', type=int, default=3, help='Number of models to ensemble')
    parser.add_argument('--models', nargs="+", type=str, default=['TimmEfficientNetB3','TimmEfficientNetB3','TimmEfficientNetB3'], help='model type (default: BaseModel)')
    parser.add_argument('--ensemble_weights', nargs="+", type=float, default=[0.33,0.33,0.33])

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dirs', nargs="+", type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dirs = args.model_dirs
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dirs, output_dir, args)
