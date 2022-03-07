import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device, num):
    if num == 1: model_cls = getattr(import_module("model"), args.model1)
    elif num == 2: model_cls = getattr(import_module("model"), args.model2)
    elif num == 3: model_cls = getattr(import_module("model"), args.model3)
    elif num == 4: model_cls = getattr(import_module("model"), args.model4)
    else: model_cls = getattr(import_module("model"), args.model5)
    
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'model{num-1}_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model1 = load_model(model_dir, num_classes, device, 1).to(device)
    model2 = load_model(model_dir, num_classes, device, 2).to(device)
    model3 = load_model(model_dir, num_classes, device, 3).to(device)
    model4 = load_model(model_dir, num_classes, device, 4).to(device)
    model5 = load_model(model_dir, num_classes, device, 5).to(device)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    check = True
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            print(f"model1 with loader {idx} predicting...")
            pred1 = model1(images)
            print(f"model2 with loader {idx} predicting...")
            pred2 = model2(images)
            print(f"model3 with loader {idx} predicting...")
            pred3 = model3(images)
            print(f"model4 with loader {idx} predicting...")
            pred4 = model4(images)
            print(f"model5 with loader {idx} predicting...")
            pred5 = model5(images)
            
            if check:
                print("pred1 : ", pred1.shape)
            pred = pred1 + pred2 + pred3 + pred4 + pred5
            if check:
                print("pred : ", pred.shape)
                check = False
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(384, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model1', type=str, default='efficientnet_b4', help='model type (default: BaseModel)')
    parser.add_argument('--model2', type=str, default='beit_base_patch16_384', help='model type (default: BaseModel)')
    parser.add_argument('--model3', type=str, default='vit_base_patch16_384', help='model type (default: BaseModel)')
    parser.add_argument('--model4', type=str, default='swin_base_patch4_window12_384', help='model type (default: BaseModel)')
    parser.add_argument('--model5', type=str, default='vit_small_r26_s32_384', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/ensenble_v1'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
