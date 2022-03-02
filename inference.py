import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from tqdm import tqdm

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
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
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=6, # default : 8
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output_tf_efficientnet_b5_ns_lr1e-5_f1.csv'), index=False)
    print(f'Inference Done!')

def CV_load_model(index, saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'best_fold{index+1}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def CV_inference(data_dir, model_dir, output_dir, args):
    """
    """
    k_fold = 5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18

    fold_preds_list = []
    for idx in range(k_fold):
        model = CV_load_model(idx, model_dir, num_classes, device).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=6, # default : 8
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print(f"Calculating fold {idx+1} inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(tqdm(loader)):
                images = images.to(device)
                pred = model(images)
                # pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())
        fold_preds_list.append(np.array(preds))

    fold_preds = np.zeros_like(fold_preds_list[0])
    for preds in fold_preds_list:
        fold_preds += preds
    fold_preds = np.argmax(fold_preds, -1)
    info['ans'] = fold_preds
    info.to_csv(os.path.join(output_dir, f'output_{args.model}_{args.learning}_lr1e-5.csv'), index=False)
    print(f'CV Inference Done!')

@torch.no_grad()
def multi_inference(data_dir, model_dir, output_dir, args):
    """
    """
    k_fold = 5
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classes = [3, 2, 3]
    features = ['age', 'gender', 'mask']
    model_dir_list = []

    total_preds_list = []

    for feature, num_class in zip(features, classes):
        print(f"-------{feature}-------")
        num_classes = num_class
        fold_preds_list = []

        model_dir_feature = model_dir + '_' + feature
        print(model_dir)
        for idx in range(k_fold):
            model = CV_load_model(idx, model_dir_feature, num_classes, device).to(device)
            model.eval()

            img_root = os.path.join(data_dir, 'images')
            info_path = os.path.join(data_dir, 'info.csv')
            info = pd.read_csv(info_path)

            img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
            dataset = TestDataset(img_paths, args.resize)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=6, # default : 8
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=False,
            )

            print(f"Calculating fold {idx+1} inference results..")
            preds = []
            with torch.no_grad():
                for idx, images in enumerate(tqdm(loader)):
                    images = images.to(device)
                    pred = model(images)
                    # pred = pred.argmax(dim=-1)
                    preds.extend(pred.cpu().numpy())
            fold_preds_list.append(np.array(preds))
        print(fold_preds_list)
        fold_preds = np.zeros_like(fold_preds_list[0])
        for preds in fold_preds_list:
            fold_preds += preds
        fold_preds = np.argmax(fold_preds, -1)
        
        if feature == 'gender':
            fold_preds *= 3
        elif feature == 'mask':
            fold_preds *= 6

        total_preds_list.append(np.array(fold_preds))

    total_preds = np.zeros_like(total_preds_list[0])
    for preds in total_preds_list:
        total_preds += preds
    info['ans'] = total_preds
    info.to_csv(os.path.join(output_dir, f'output_{args.model}_{args.learning}_lr1e-5.csv'), index=False)
    print(f'multi Inference Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=(384, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--learning', type=str, default='single', help='learning method (default: single)')


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/code/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.learning == "CV":
        CV_inference(data_dir, model_dir, output_dir, args)
    elif args.learning == "multi":
        multi_inference(data_dir, model_dir, output_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)
