import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import yaml
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.dataset import CustomDataset
from utils.metrics import evaluate_model
from utils.transforms import get_transforms
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """自定义批次整理函数"""
    return tuple(zip(*batch))


def create_model(config):
    """创建Faster R-CNN模型"""
    model_config = config['model']
    backbone = model_config.get('backbone', 'resnet50')
    pretrained = model_config.get('pretrained', True)
    num_classes = model_config['num_classes']  # 包含背景类

    # 根据backbone选择模型
    if backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif backbone == 'mobilenet':
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的backbone: {backbone}")

    # 替换分类器头部
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def create_optimizer(model, config):
    """创建优化器"""
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type']
    lr = opt_config['lr']

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == 'SGD':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0.0005)
        )
    elif opt_type == 'Adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0005)
        )
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0005)
        )
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")

    return optimizer


def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']

    if scheduler_type == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config['milestones'],
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控mAP，越大越好
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    return scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger):
    """训练一个epoch"""
    model.train()
    header = f'Epoch: [{epoch}]'

    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0

    # 使用 tqdm 包装 data_loader
    progress_bar = tqdm(data_loader, desc=header, leave=True, ncols=120) # leave=True 可以在循环结束后保留进度条

    for i, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 统计损失
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 创建一个包含所有损失的字典
        postfix_dict = {
            'loss': f'{losses.item():.3f}',
            'cls': f'{loss_dict["loss_classifier"].item():.3f}',
            'box': f'{loss_dict["loss_box_reg"].item():.3f}',
            'obj': f'{loss_dict["loss_objectness"].item():.3f}',
            'rpn': f'{loss_dict["loss_rpn_box_reg"].item():.3f}'
        }
        # 将字典设置到进度条的后缀中
        progress_bar.set_postfix(postfix_dict)

    # 计算平均损失
    num_batches = len(data_loader)
    # 保留小数点后两位
    avg_losses = {
        'total_loss': round(total_loss / num_batches, 4),
        'loss_classifier': round(total_loss_classifier / num_batches, 4),
        'loss_box_reg': round(total_loss_box_reg / num_batches, 4),
        'loss_objectness': round(total_loss_objectness / num_batches, 4),
        'loss_rpn_box_reg': round(total_loss_rpn_box_reg / num_batches, 4)
    }

    return avg_losses


def main():
    args = parse_args()

    # 加载配置文件
    config = load_config(args.config)
    print(f"加载配置文件: {args.config}")

    # 从配置文件中获取参数
    dataset_config = config['dataset']
    training_config = config['training']
    validation_config = config['validation']
    saving_config = config['saving']
    logging_config = config['logging']

    # 创建保存目录
    save_dir = Path(f"{saving_config['save_dir']}/{dataset_config['name']}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('train', f"{logging_config['log_dir']}/{dataset_config['name']}_train.log")
    logger.info(f'训练配置: {config}')

    # 设置设备
    device = torch.device(training_config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 创建数据集和数据加载器
    train_transforms = get_transforms(train=True, config=config)
    val_transforms = get_transforms(train=False, config=config)

    train_dataset = CustomDataset(
        root=dataset_config['root'],
        split='train',
        transforms=train_transforms,
        format=dataset_config['format']
    )

    val_dataset = CustomDataset(
        root=dataset_config['root'],
        split='val',
        transforms=val_transforms,
        format=dataset_config['format']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training_config['num_workers'],
        collate_fn=collate_fn
    )

    logger.info(f'训练集大小: {len(train_dataset)}')
    logger.info(f'验证集大小: {len(val_dataset)}')

    # 创建模型
    model = create_model(config)
    model.to(device)
    logger.info(f"模型backbone: {config['model']['backbone']}")

    # 创建优化器和学习率调度器
    optimizer = create_optimizer(model, config)
    lr_scheduler = create_scheduler(optimizer, config)

    # 恢复训练
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)
        logger.info(f'从epoch {start_epoch}恢复训练, 最佳mAP: {best_map:.4f}')
    
    # 训练循环
    logger.info('开始训练...')
    for epoch in range(start_epoch, training_config['epochs']):
        # 训练一个epoch
        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch, logger
        )

        # 更新学习率
        if config['training']['scheduler']['type'] == 'ReduceLROnPlateau':
            # ReduceLROnPlateau需要在评估后更新
            pass
        else:
            lr_scheduler.step()

        # 记录训练损失
        logger.info(f'Epoch {epoch} 训练损失: {train_losses}')

        # 评估模型
        if (epoch + 1) % validation_config['eval_freq'] == 0:
            logger.info('开始评估...')
            eval_results = evaluate_model(model, val_loader, device)
            current_map = eval_results['map_50']

            logger.info(f'Epoch {epoch} 评估结果: mAP@0.5:0.95={eval_results["map_50_95"]:.4f}, mAP@0.5={eval_results["map_50"]:.4f}, map@0.75={eval_results["map_75"]:.4f}, ar@0.5:0.95(max=100)={eval_results["ar_100"]:.4f}')

            # 如果使用ReduceLROnPlateau，在这里更新学习率
            if config['training']['scheduler']['type'] == 'ReduceLROnPlateau':
                lr_scheduler.step(current_map)

            # 保存最佳模型
            is_best = current_map > best_map
            if is_best:
                best_map = current_map
                logger.info(f'新的最佳模型! mAP@0.5: {best_map:.4f}')

            # 保存检查点
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_map': best_map,
                'config': config
            }, is_best, save_dir)

        # 定期保存模型
        if (epoch + 1) % saving_config['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_map': best_map,
                'config': config
            }, False, save_dir, f'checkpoint_epoch_{epoch + 1}.pth')

    logger.info('训练完成!')
    logger.info(f'最佳mAP@0.5: {best_map:.4f}')


if __name__ == '__main__':
    main()