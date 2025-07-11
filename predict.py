"""
Faster R-CNN推理脚本
"""
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.transforms import get_transforms
from utils.visualization import draw_boxes, save_prediction_image
from train import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='runs/predict',
                        help='输出目录')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='类别数量(包含背景)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        help='类别名称列表')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存预测结果为txt格式')
    parser.add_argument('--save-json', action='store_true',
                        help='保存预测结果为json格式')
    parser.add_argument('--save-image', action='store_true',
                        help='保存可视化图像')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    return parser.parse_args()


def load_image(image_path):
    """加载图像"""
    image = Image.open(image_path).convert('RGB')
    return image


def preprocess_image(image, transforms):
    """预处理图像"""
    if transforms:
        image, _ = transforms(image, {})
    return image


@torch.no_grad()
def predict_single_image(model, image, device, conf_threshold=0.5):
    """对单张图像进行预测"""
    model.eval()

    # 预处理
    if isinstance(image, Image.Image):
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    else:
        image_tensor = image

    image_tensor = image_tensor.to(device)

    # 推理
    predictions = model([image_tensor])
    prediction = predictions[0]

    # 过滤低置信度预测
    keep = prediction['scores'] >= conf_threshold
    boxes = prediction['boxes'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()

    return boxes, labels, scores


def save_results_txt(boxes, labels, scores, output_path, img_width, img_height):
    """保存结果为txt格式 (YOLO格式)"""
    with open(output_path, 'w') as f:
        for box, label, score in zip(boxes, labels, scores):
            # 转换为YOLO格式
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            f.write(f"{label - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")


def save_results_json(boxes, labels, scores, output_path, class_names=None):
    """保存结果为json格式"""
    results = []
    for box, label, score in zip(boxes, labels, scores):
        result = {
            'bbox': box.tolist(),
            'category_id': int(label),
            'category_name': class_names[label - 1] if class_names and label > 0 else f'class_{label}',
            'score': float(score)
        }
        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def process_single_image(model, image_path, args, device, transforms):
    """处理单张图像"""
    # 加载图像
    image = load_image(image_path)
    original_size = image.size  # (width, height)

    # 预处理
    image_tensor = preprocess_image(image, transforms)

    # 预测
    start_time = time.time()
    boxes, labels, scores = predict_single_image(
        model, image_tensor, device, args.conf_threshold
    )
    inference_time = time.time() - start_time

    print(f"图像: {image_path.name}")
    print(f"推理时间: {inference_time:.4f}秒")
    print(f"检测到 {len(boxes)} 个目标")

    # 创建输出路径
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # 保存txt结果
    if args.save_txt:
        txt_path = output_dir / f"{stem}.txt"
        save_results_txt(boxes, labels, scores, txt_path, *original_size)

    # 保存json结果
    if args.save_json:
        json_path = output_dir / f"{stem}.json"
        save_results_json(boxes, labels, scores, json_path, args.class_names)

    # 保存可视化图像
    if args.save_image:
        image_path_out = output_dir / f"{stem}_pred.jpg"
        save_prediction_image(
            image, boxes, labels, scores, image_path_out, args.class_names
        )

    # 显示结果
    if args.show:
        visualize_prediction(image, boxes, labels, scores, args.class_names)

    return boxes, labels, scores


def visualize_prediction(image, boxes, labels, scores, class_names=None):
    """可视化预测结果"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # 绘制边框
        color = colors[label % len(colors)]
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # 添加标签
        if class_names and label > 0 and label - 1 < len(class_names):
            class_name = class_names[label - 1]
        else:
            class_name = f'Class {label}'

        ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=10, color='black')

    ax.axis('off')
    plt.title(f'检测结果 - {len(boxes)} 个目标')
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

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
    model.eval()
    print(f'模型加载成功: {args.model_path}')

    # 获取预处理变换
    transforms = get_transforms(train=False)

    # 处理输入
    input_path = Path(args.input)

    if input_path.is_file():
        # 单张图像
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_single_image(model, input_path, args, device, transforms)
        else:
            print(f"不支持的图像格式: {input_path.suffix}")

    elif input_path.is_dir():
        # 图像目录
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"在目录 {input_path} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像")

        total_time = 0
        for image_file in image_files:
            start_time = time.time()
            process_single_image(model, image_file, args, device, transforms)
            total_time += time.time() - start_time

        print(f"\n总处理时间: {total_time:.2f}秒")
        print(f"平均每张图像: {total_time / len(image_files):.4f}秒")

    else:
        print(f"输入路径不存在: {input_path}")


if __name__ == '__main__':
    main()