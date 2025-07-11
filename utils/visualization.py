"""
可视化工具
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns


def draw_boxes(image, boxes, labels, scores, class_names=None, score_threshold=0.5):
    """在图像上绘制边框"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    image = image.copy()

    # 定义颜色
    colors = plt.cm.Set3(np.linspace(0, 1, max(labels) + 1))

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        # 获取颜色
        color = colors[label][:3]
        color = tuple(int(c * 255) for c in color)

        # 绘制边框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        if class_names and label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f'Class {label}'

        text = f'{class_name}: {score:.2f}'

        # 计算文本尺寸
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # 绘制文本背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return image


def save_prediction_image(image, boxes, labels, scores, save_path, class_names=None):
    """保存预测结果图像"""
    result_image = draw_boxes(image, boxes, labels, scores, class_names)

    if isinstance(result_image, np.ndarray):
        result_image = Image.fromarray(result_image)

    result_image.save(save_path)


def plot_training_curves(train_losses, val_maps, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 训练损失
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 验证mAP
    if val_maps:
        val_epochs = range(1, len(val_maps) + 1)
        ax2.plot(val_epochs, val_maps, 'r-', label='Validation mAP')
        ax2.set_title('Validation mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_dataset_samples(dataset, num_samples=6, class_names=None):
    """可视化数据集样本"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        if i >= 6:
            break

        image, target = dataset[idx]

        # 转换图像格式
        if isinstance(image, torch.Tensor):
            # 反标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image.permute(1, 2, 0).numpy()
            image = image * std + mean
            image = np.clip(image, 0, 1)

        # 绘制图像和边框
        axes[i].imshow(image)

        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 绘制边框
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[i].add_patch(rect)

            # 添加标签
            if class_names and label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f'Class {label}'

            axes[i].text(
                x1, y1 - 5, class_name,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                fontsize=8, color='white'
            )

        axes[i].set_title(f'Sample {idx}')
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(indices), 6):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataset, class_names=None, save_path=None):
    """绘制类别分布"""
    class_counts = {}

    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels'].numpy()

        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    # 准备数据
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]

    if class_names:
        class_labels = [class_names[cls] if cls < len(class_names) else f'Class {cls}' for cls in classes]
    else:
        class_labels = [f'Class {cls}' for cls in classes]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_confusion_matrix(predictions, targets, num_classes, class_names=None):
    """创建混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import itertools

    # 收集所有预测和真实标签
    all_preds = []
    all_targets = []

    for pred, target in zip(predictions, targets):
        pred_labels = pred['labels'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()

        all_preds.extend(pred_labels)
        all_targets.extend(target_labels)

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    # 添加数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return cm


def plot_detection_results_grid(images, predictions, targets=None, class_names=None,
                                score_threshold=0.5, save_path=None):
    """网格显示检测结果"""
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_images):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # 显示图像
        if isinstance(images[i], torch.Tensor):
            img = images[i].permute(1, 2, 0).cpu().numpy()
        else:
            img = np.array(images[i])

        ax.imshow(img)

        # 绘制预测边框
        pred = predictions[i]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle((x1, y1), width, height,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                if class_names and label < len(class_names):
                    class_name = class_names[label]
                else:
                    class_name = f'C{label}'

                ax.text(x1, y1 - 5, f'{class_name}:{score:.2f}',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        fontsize=8, color='white')

        # 绘制真实边框（如果提供）
        if targets:
            target = targets[i]
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()

            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle((x1, y1), width, height,
                                         linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                ax.add_patch(rect)

        ax.set_title(f'Image {i + 1}')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()