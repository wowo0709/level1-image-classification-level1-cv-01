import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from tqdm import tqdm
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from early_stopping import EarlyStopping

from sklearn.metrics import f1_score

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


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


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

def cross_valid_train(data_dir, model_dir, args, Ldam_cls):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        args = args,
        k_fold = 5,
        features = args.features
    )
    num_classes = args.num_classes  # 18 or (3 2 3)

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: AlbuAugmentation
    transform_train = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    transform_val_module = getattr(import_module("dataset"), "AlbuAugmentationVal")
    transform_val = transform_val_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    # -- data_loader
    train_val_set = dataset.split_dataset()
    best_f1_score_list = []
    for set_idx, (train_set, val_set) in enumerate(train_val_set):

        print(f"----------fold {set_idx+1} start----------")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)



        # -- loss & metric
        criterion = create_criterion(args.criterion, Ldam_cls[args.features])  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )

        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr=1e-5,
        #     max_lr=args.lr,
        #     step_size_down=len(train_set) * 2 // args.batch_size,
        #     step_size_up=len(train_set) // args.batch_size,
        #     cycle_momentum=False,
        #     mode="triangular2"
        #     )

        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        best_f1_score = 0
        early_stop = EarlyStopping(name=args.name)
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            # train loop
            model.train()
            dataset.set_transform(transform_train)

            loss_value = 0
            matches = 0
            temp_loss_value = 0
            temp_matches = 0
            y_true, y_pred = [], []

            for idx, train_batch in enumerate(tqdm(train_loader)):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                temp_loss_value += loss.item()
                matches += (preds == labels).sum().item()
                temp_matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    temp_train_loss = temp_loss_value / args.log_interval
                    temp_train_acc = temp_matches / args.batch_size / args.log_interval

                    logger.add_scalar("Train/loss", temp_train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", temp_train_acc, epoch * len(train_loader) + idx)

                    temp_loss_value = 0
                    temp_matches = 0

            train_loss = loss_value / len(train_loader)
            train_acc = matches / (args.batch_size *len(train_loader))
            f1 = f1_score(y_pred, y_true, average='macro')
            current_lr = get_lr(optimizer)
            print(
                f"Epoch[{epoch+1}/{args.epochs}] || F1 score {f1:4.4} || "
                f"training accuracy {train_acc:4.2%} || training loss {train_loss:4.4} || lr {current_lr} || "
            )
            scheduler.step()
            torch.cuda.empty_cache()

            # val loop
            dataset.set_transform(transform_val)
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                y_true, y_pred = [], []

                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    y_true.extend(labels.tolist())
                    y_pred.extend(preds.tolist())

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                f1 = f1_score(y_pred, y_true, average='macro')

                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                if f1 > best_f1_score: #val_acc > best_val_acc and val_loss < best_val_loss:
                    print(f"----New best model for val f1 score : {f1:4.4}! saving the best model..----")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_fold{set_idx+1}.pth")
                    best_f1_score = f1
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] || F1 score : {f1:4.4}, acc : {val_acc:4.2%}, loss: {val_loss:4.3} || \n"
                    f"[Val Best] best F1 score {best_f1_score:4.4}, best acc : {best_val_acc:4.3%}, best loss: {best_val_loss:4.2}"
                )

                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/F1", f1, epoch)
                logger.add_figure("results", figure, epoch)

            if early_stop(val_loss, model):
                print("early stop!!!")
                print()
                break
            print()
        best_f1_score_list.append(best_f1_score)
    print('>> Cross Validation Finish')
    print(f'CV F1-Score: {np.mean(best_f1_score_list)}')

def multi_train(data_dir, model_dir, args):
    features = ['age', 'gender', 'mask']
    criterions = ['LDAM', 'LDAM', 'cross_entropy']
    classes = [3, 2, 3]
    Ldam_cls = {
        "age" : [1281, 983, 436],
        "gender" : [1041, 1658],
        "mask" : []
    }

    for feature, criterion, num_classes in zip(features, criterions, classes):
        print(f"-----{feature}-----")
        args.criterion = criterion
        args.num_classes = num_classes
        args.name = args.name+'_'+feature
        args.features = feature

        cross_valid_train(data_dir, model_dir, args, Ldam_cls)

        args.name = "_".join(args.name.split('_')[:-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='CrossValid', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='AlbuAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[384, 384], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    parser.add_argument('--features', default='False', help='given in multi label')
    parser.add_argument('--num_classes', default=18, help='num_classes')
    parser.add_argument('--multi', default=0, help='method train')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/face_images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/code/model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    if args.multi:
        multi_train(data_dir, model_dir, args)
    else:
        cross_valid_train(data_dir, model_dir, args)