import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import logging
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SplitByMaskDataset, SplitByGenderDataset, SplitByAgeDataset
from loss import create_criterion
from utils import EarlyStopping

wandb.init(project="level 1-p stage", entity="wowo0709")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# def grid_image(np_images, gts, preds, n=16, shuffle=False):
#     batch_size = np_images.shape[0]
#     logging.info(f"n: {n} batch_size: {batch_size}")
#     assert n <= batch_size

#     choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
#     figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
#     plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
#     n_grid = np.ceil(n ** 0.5)
#     tasks = ["mask", "gender", "age"]
#     for idx, choice in enumerate(choices):
#         gt = gts[choice].item()
#         pred = preds[choice].item()
#         image = np_images[choice]
#         # title = f"gt: {gt}, pred: {pred}"
#         gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
#         pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
#         title = "\n".join([
#             f"{task} - gt: {gt_label}, pred: {pred_label}"
#             for gt_label, pred_label, task
#             in zip(gt_decoded_labels, pred_decoded_labels, tasks)
#         ])

#         plt.subplot(n_grid, n_grid, idx + 1, title=title)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(image, cmap=plt.cm.binary)

#     return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    # dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseMaskDataset
    # dataset = dataset_module(
    #     data_dir=data_dir,
    # )
    mask_dataset = SplitByMaskDataset(data_dir=data_dir)
    gender_dataset = SplitByGenderDataset(data_dir=data_dir)
    age_dataset = SplitByAgeDataset(data_dir=data_dir)
    mask_num_classes = mask_dataset.num_classes  # 3
    gender_num_classes = gender_dataset.num_classes  # 2
    age_num_classes = age_dataset.num_classes  # 3

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    mask_transform = transform_module(
        resize=args.resize,
        mean=mask_dataset.mean,
        std=mask_dataset.std,
    )
    gender_transform = transform_module(
        resize=args.resize,
        mean=gender_dataset.mean,
        std=gender_dataset.std,
    )
    age_transform = transform_module(
        resize=args.resize,
        mean=age_dataset.mean,
        std=age_dataset.std,
    )
    mask_dataset.set_transform(mask_transform)
    gender_dataset.set_transform(gender_transform)
    age_dataset.set_transform(age_transform)

    # -- data_loader
    mask_train_set, mask_val_set = mask_dataset.split_dataset()
    gender_train_set, gender_val_set = gender_dataset.split_dataset()
    age_train_set, age_val_set = age_dataset.split_dataset()

    mask_train_loader = DataLoader(
        mask_train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    gender_train_loader = DataLoader(
        gender_train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    age_train_loader = DataLoader(
        age_train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    mask_val_loader = DataLoader(
        mask_val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    gender_val_loader = DataLoader(
        gender_val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    age_val_loader = DataLoader(
        age_val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    mask_model = model_module(
        num_classes=mask_num_classes
    ).to(device)
    gender_model = model_module(
        num_classes=gender_num_classes
    ).to(device)
    age_model = model_module(
        num_classes=age_num_classes
    ).to(device)

    mask_model = torch.nn.DataParallel(mask_model)
    gender_model = torch.nn.DataParallel(gender_model)
    age_model = torch.nn.DataParallel(age_model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD

    mask_optimizer = opt_module(
        filter(lambda p: p.requires_grad, mask_model.parameters()),
        lr=args.lr,
        weight_decay=1e-3 # 5e-4
    )
    gender_optimizer = opt_module(
        filter(lambda p: p.requires_grad, gender_model.parameters()),
        lr=args.lr,
        weight_decay=1e-3 # 5e-4
    )
    age_optimizer = opt_module(
        filter(lambda p: p.requires_grad, age_model.parameters()),
        lr=args.lr,
        weight_decay=1e-3 # 5e-4
    )

    mask_scheduler = StepLR(mask_optimizer, args.lr_decay_step, gamma=0.5)
    gender_scheduler = StepLR(mask_optimizer, args.lr_decay_step, gamma=0.5)
    age_scheduler = StepLR(mask_optimizer, args.lr_decay_step, gamma=0.5)

    # -- compile options
    mask_early_stopping = EarlyStopping(patience=7, verbose=True, path=os.path.join(save_dir, 'mask_early_stopping.pth'))
    gender_early_stopping = EarlyStopping(patience=7, verbose=True, path=os.path.join(save_dir, 'gender_early_stopping.pth'))
    age_early_stopping = EarlyStopping(patience=7, verbose=True, path=os.path.join(save_dir, 'age_early_stopping.pth'))

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
        # wandb logging
        wandb.config = f

    # -- phase
    mask_best_val_acc, gender_best_val_acc, age_best_val_acc = 0, 0, 0
    mask_best_val_loss, gender_best_val_loss, age_best_val_loss = np.inf, np.inf, np.inf

    phases = {
        'mask': {
            'model': mask_model,
            'train_set': mask_train_set,
            'val_set': mask_val_set,
            'train_loader': mask_train_loader,
            'val_loader': mask_val_loader, 
            'optimizer': mask_optimizer, 
            'scheduler': mask_scheduler,
            'early_stopping': mask_early_stopping,
            'best_val_acc': mask_best_val_acc,
            'best_val_loss': mask_best_val_loss
        },
        'gender': {
            'model': gender_model,
            'train_set': gender_train_set,
            'val_set': gender_val_set,
            'train_loader': gender_train_loader,
            'val_loader': gender_val_loader, 
            'optimizer': gender_optimizer, 
            'scheduler': gender_scheduler,
            'early_stopping': gender_early_stopping,
            'best_val_acc': gender_best_val_acc,
            'best_val_loss': gender_best_val_loss
        },
        'age': {
            'model': age_model,
            'train_set': age_train_set,
            'val_set': age_val_set,
            'train_loader': age_train_loader,
            'val_loader': age_val_loader,
            'optimizer': age_optimizer,
            'scheduler': age_scheduler,
            'early_stopping': age_early_stopping,
            'best_val_acc': age_best_val_acc,
            'best_val_loss': age_best_val_loss
        }
    }

    # -- training
    for epoch in range(args.epochs):
        for phase in phases.keys():
            print(f"Current phase is '{phase}'")
            model, train_set, val_set, train_loader, val_loader, optimizer, scheduler, early_stopping, best_val_acc, best_val_loss = phases[phase].values()
            if early_stopping.early_stop:
                print(f"{phase} - Early Stopping")
                continue
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"{phase} - Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar(f"{phase}/Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar(f"{phase}/Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # weights&biases logging - train
            wandb.log({f'{phase}_train_accuracy': train_acc, f'{phase}_train_loss': train_loss})

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    # if figure is None:
                    #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    #     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    #     figure = grid_image(
                    #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    #     )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best {phase} model for val accuracy : {val_acc:4.2%}! saving the best {phase} model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/{phase}_best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/{phase}_last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar(f"{phase}/Val/loss", val_loss, epoch)
                logger.add_scalar(f"{phase}/Val/accuracy", val_acc, epoch)
                # logger.add_figure("results", figure, epoch)
                print()

            # weights&biases logging - validation
            wandb.log({f'{phase}_val_accuracy': val_acc, f'{phase}_val_loss': val_loss})

            # early stopping
            early_stopping(val_loss, model)

            phases[phase]['model'] = model
            phases[phase]['train_set'] = train_set
            phases[phase]['val_set'] = val_set
            phases[phase]['train_loader'] = train_loader
            phases[phase]['val_loader'] = val_loader
            phases[phase]['optimizer'] = optimizer
            phases[phase]['scheduler'] = scheduler
            phases[phase]['early_stopping'] = early_stopping
            phases[phase]['best_val_acc'] = best_val_acc
            phases[phase]['best_val_loss'] = best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoints'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
