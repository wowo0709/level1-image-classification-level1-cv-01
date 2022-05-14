import argparse
from distutils.command.config import config
from gc import callbacks
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
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from utils import EarlyStopping

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import WandbLoggerCallback, wandb_mixin


# wandb.init(project="level 1-p stage-ray tune", entity="wowo0709")


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

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_dir = increment_path(os.path.join(model_dir, args.name))
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    '''
    종속 변인: Dataset의 Accuracy의 최대화
    조작 변인: Epoch, Batch size, Learning rate, Loss, Model
    통제 변인: Dataset, Scheduler, Augmentation, Optimizer
    '''


    # -- epoch => 조작 변인
    def get_epoch_by_epoch(epoch:int):
        return epoch

    # 
    # # -- dataset => 통제 변인
    # dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseMaskDataset
    # dataset = dataset_module(
    #     data_dir=data_dir,
    # )
    # num_classes = dataset.num_classes  # 18


    # # -- augmentation => 통제 변인
    # transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    # dataset.set_transform(transform)

    # train_set, val_set = dataset.split_dataset()


    # -- data_loader => 조작 변인
    def get_dataloaders_by_batchsize(train_set, val_set, batch_size:int):
        BATCH_SIZE = batch_size
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        return train_loader, val_loader


    # -- model => 조작 변인
    def get_model_by_model_name(model_name, num_classes):
        model_module = getattr(import_module("model"), model_name)
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        return model


    # -- loss & metric
    def get_criterion_by_criterion_name(criterion_name):
        criterion = create_criterion(criterion_name)

        return criterion


    def get_optimizer_and_scheduler_by_lr(model, learning_rate:float):
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-3 # 5e-4
        )
        sch_module = getattr(import_module("torch.optim.lr_scheduler"), args.scheduler)
        if args.scheduler == 'CyclicLR':
            scheduler = sch_module(
                optimizer, 
                base_lr=1e-6,
                max_lr=learning_rate,
                step_size_up=args.lr_decay_step,
                mode="exp_range",
                gamma=0.7,
                cycle_momentum=False
            )

        return optimizer, scheduler



    # -- 탐색할 hyperparameter config 설정
    # 조작 변인: Epoch, Batch size, Learning rate, Loss, Model, Scheduler
    config_space = {
        "NUM_EPOCH": tune.choice(args.epochs),
        "BATCH_SIZE": tune.choice(args.batch_sizes),
        "LEARNING_RATE": tune.choice(args.lrs),# tune.uniform(1e-4,1e-5),
        "CRITERION": tune.choice(args.criterions),
        "MODEL": tune.choice(args.models)
    }
    # 탐색할 Optimizer 설정
    hpo = HyperOptSearch(
        metric='accuracy',
        mode='max'
    )

    # Training 함수 작성
    '''
    종속 변인: Dataset의 Accuracy의 최대화
    조작 변인: Epoch, Batch size, Learning rate, Loss, Model
    통제 변인: Dataset, Scheduler, Augmentation, Optimizer
    '''
    # @wandb_mixin
    def training_fn(config, checkpoint_dir=None):
        wandb.init()

        # -- ray tune에서 알아서 save
        # save_dir = increment_path(os.path.join(model_dir, args.name))
        # tm = time.localtime(time.time())
        # identifier = '_'.join(list(map(str,[tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec])))
        # log_dir = os.path.join(save_dir, identifier)

        # -- ray tune에서 알아서 stop
        # -- compile options
        # early_stopping = EarlyStopping(patience=5, verbose=True, path=os.path.join(save_dir, 'early_stopping.pth'))

        # 통제 변인
        # -- dataset => 통제 변인
        dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseMaskDataset
        dataset = dataset_module(
            data_dir=data_dir,
        )
        num_classes = dataset.num_classes  # 18


        # -- augmentation => 통제 변인
        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)

        train_set, val_set = dataset.split_dataset()

        # 조작 변인
        epochs = get_epoch_by_epoch(config["NUM_EPOCH"])
        batch_size = config["BATCH_SIZE"]
        train_loader, val_loader = get_dataloaders_by_batchsize(train_set, val_set, batch_size)
        criterion_name = config["CRITERION"]
        criterion = get_criterion_by_criterion_name(criterion_name)
        if checkpoint_dir:
            model = torch.load(os.path.join(checkpoint_dir, "best.pth"))
        else:
            model_name = config["MODEL"]
            model = get_model_by_model_name(model_name, num_classes)
        lr = config["LEARNING_RATE"]
        optimizer, scheduler = get_optimizer_and_scheduler_by_lr(model, lr)
            

        # -- ray tune에서 알아서 save
        # # -- logging
        # hps = {
        #     'epoch': epochs,
        #     'batch_size': batch_size,
        #     'criterion': criterion_name, 
        #     'model': model_name, 
        #     'lr': lr
        # }
        # with open(os.path.join(log_dir, 'config.json'), 'w', encoding='utf-8') as f:
        #     json.dump(hps, f, ensure_ascii=False, indent=4)
        #     # wandb logging
        #     wandb.config = f


        # training
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(epochs):
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
                    train_acc = matches / batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch+1}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # weights&biases logging - train
            wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss})

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                # figure = None
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
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    # torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    with tune.checkpoint_dir(epochs) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "best.pth")
                        torch.save(model.state_dict(), path)
                    best_val_acc = val_acc
                # torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                with tune.checkpoint_dir(epochs) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "last.pth")
                    torch.save(model.state_dict(), path)
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                # logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                # logger.add_figure("results", figure, epoch)
                print()

            # weights&biases logging - validation
            wandb.log({'val_accuracy': val_acc, 'val_loss': val_loss})

            # # early stopping
            # early_stopping(val_loss, model)
            # if early_stopping.early_stop:
            #     print("Early Stopping")
            #     break


        tune.report(accuracy=best_val_acc, loss=best_val_loss)


    # Tuning 수행
    NUM_TRIAL = args.num_trial # Hyper Parameter를 탐색할 때 실험을 최대 수행할 횟수를 지정

    reporter = CLIReporter( # 중간 수행 결과를 command line에 출력
        parameter_columns=list(config_space.keys()),
        metric_columns=["accuracy", "loss"]
    )

    scheduler = ASHAScheduler(metric="accuracy", mode="max")

    ray.shutdown() # ray 초기화 후 실행
    
    analysis = tune.run(
        partial(training_fn,checkpoint_dir=None),
        config=config_space,
        search_alg=hpo,
        verbose=1,
        progress_reporter=reporter,
        scheduler=scheduler,
        num_samples=NUM_TRIAL,
        resources_per_trial={'gpu': 1}, # GPU를 사용하지 않는다면 comment 처리로 지워주세요
        local_dir="/opt/ml/checkpoints", # save directory path
        name=args.name, # experiment name
        # checkpoint_at_end=True,
        callbacks=[WandbLoggerCallback(
            project="level 1-p stage-ray tune",
            api_key_file='/opt/ml/wandb/api_key_file',
            entity="wowo0709",
            log_config=True
        )]
    )


    # 결과 확인
    best_trial = analysis.get_best_trial('accuracy', 'max')
    print(f"최고 성능 config : {best_trial.config}")
    print(f"최고 val accuracy : {best_trial.last_result['accuracy']}")
    print(f"최저 val loss: {best_trial.last_result['loss']}")
    print(f"Best checkpoint directory: {best_trial.checkpoint}")












if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='optimizer scheduler type (default: StepLR)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    # ray tune =================================================================================
    '''
    config_space = {
        "NUM_EPOCH": tune.choice([5,10,15]),
        "BATCH_SIZE": tune.choice([8,16,32]),
        "LEARNING_RATE": tune.choice([1e-4,1e-5]),# tune.uniform(1e-4,1e-5),
        "CRITERION": tune.choice(args.criterions),
        "MODEL": tune.choice(args.models)
    }
    '''
    parser.add_argument('--num_trial', type=int, default=2)
    parser.add_argument('--epochs', nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument('--batch_sizes', nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument('--lrs', nargs="+", type=float, default=[1e-4, 1e-5])
    parser.add_argument('--models', nargs="+", type=str, default=['TimmEfficientNetB4', 'TimmSwinBasePatch4Window12_384', 'TimmSwinLargePatch4Window12_384'])
    parser.add_argument('--criterions', nargs="+", type=str, default=['focal', 'ldam', 'custom_ldam'])



    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoints'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
