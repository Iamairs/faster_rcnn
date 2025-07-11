"""
Faster R-CNN测试脚本
"""
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import json
import time

from utils.dataset import CustomDataset
from utils.transforms import get_transforms
from utils.metrics import evaluate_model, CocoEvaluator
from utils.logger import setup_logger
from train import create_model, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Testing')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='类别数量(包含背景)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='测试数据集分割')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS阈值')
    parser.add_argument('--device', type=str, default='cuda',
                        help='测试设备')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--save-results', action='store_true',
                        help='保存预测结果')
    parser.add_argument('--results-dir', type=str, default='runs/test',
                        help='结果保存目录')
    return parser.parse_args()


@torch.no_grad()
def test_model(model, data_loader, device, conf_threshold=0.5, save_results=False, results_dir=None):
    """测试模型"""
    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 推理
        predictions = model(images)

        # 过滤低置信度预测
        for pred in predictions:
            keep = pred['scores'] >= conf_threshold
            pred['boxes'] = pred['boxes'][keep]
            pred['labels'] = pred['labels'][keep]
            pred['scores'] = pred['scores'][keep]

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # 保存结果
    if save_results and results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # 保存预测结果
        pred_results = []
        for i, pred in enumerate(all_predictions):
            pred_result = {
                'image_id': i,
                'boxes': pred['boxes'].cpu().numpy().tolist(),
                'labels': pred['labels'].cpu().numpy().tolist(),
                'scores': pred['scores'].cpu().numpy().tolist()
            }
            pred_results.append(pred_result)

        with open(results_path / 'predictions.json', 'w') as f:
            json.dump(pred_results, f, indent=2)

    return all_predictions, all_targets


def main():
    args = parse_args()

    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('test', results_dir / 'test.log')
    logger.info(f'测试参数: {args}')

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 创建数据集和数据加载器
    test_transforms = get_transforms(train=False)

    test_dataset = CustomDataset(
        root=args.data_root,
        split=args.split,
        transforms=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    logger.info(f'测试集大小: {len(test_dataset)}')

    # 加载模型
    model = create_model(args.num_classes, pretrained=False)

    if args.model_path.endswith('.pth'):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    model.to(device)
    logger.info(f'模型加载成功: {args.model_path}')

    # 测试模型
    logger.info('开始测试...')
    start_time = time.time()

    predictions, targets = test_model(
        model, test_loader, device,
        args.conf_threshold, args.save_results, results_dir
    )

    test_time = time.time() - start_time
    logger.info(f'测试完成, 耗时: {test_time:.2f}秒')

    # 评估模型
    logger.info('开始评估...')
    eval_results = evaluate_model(model, test_loader, device)

    # 打印结果
    logger.info('=== 测试结果 ===')
    logger.info(f"mAP@0.5: {eval_results['map_50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {eval_results['map_50_95']:.4f}")
    logger.info(f"mAP@0.75: {eval_results['map_75']:.4f}")
    logger.info(f"mAP(small): {eval_results['map_small']:.4f}")
    logger.info(f"mAP(medium): {eval_results['map_medium']:.4f}")
    logger.info(f"mAP(large): {eval_results['map_large']:.4f}")

    # 保存评估结果
    if args.save_results:
        with open(results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f'结果已保存到: {results_dir}')


if __name__ == '__main__':
    main()