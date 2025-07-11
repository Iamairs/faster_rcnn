"""
自定义数据集类
"""
import os
import shutil
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import xml.etree.ElementTree as ET
from pathlib import Path


class CustomDataset(Dataset):
    """
    自定义目标检测数据集
    支持COCO格式和VOC格式
    """

    def __init__(self, root, split='train', transforms=None, format='coco'):
        """
        Args:
            root: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            transforms: 数据变换
            format: 标注格式 ('coco', 'voc')
        """
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.format = format

        # 设置路径
        self.images_dir = self.root / split / 'images'
        self.annotations_dir = self.root / split / 'annotations'

        # 加载数据
        if format == 'coco':
            self.load_coco_data()
        elif format == 'voc':
            self.load_voc_data()
        else:
            raise ValueError(f"不支持的格式: {format}")

    def load_coco_data(self):
        """加载COCO格式数据"""
        annotation_file = self.annotations_dir / f'{self.split}.json'

        if not annotation_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {annotation_file}")

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # 创建图像ID到图像信息的映射
        self.images = {img['id']: img for img in self.coco_data['images']}

        # 创建类别ID到类别名称的映射
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # 按图像分组标注
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # 获取所有图像ID
        self.image_ids = list(self.images.keys())

    def load_voc_data(self):
        """加载VOC格式数据"""
        # 获取所有XML文件
        xml_files = list(self.annotations_dir.glob('*.xml'))

        self.voc_data = []
        self.categories = set()

        for xml_file in xml_files:
            # 解析XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取图像信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # 获取标注信息
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                self.categories.add(name)

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                objects.append({
                    'name': name,
                    'bbox': [xmin, ymin, xmax, ymax]
                })

            self.voc_data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            })

        # 创建类别到ID的映射
        self.categories = {name: idx + 1 for idx, name in enumerate(sorted(self.categories))}
        self.category_names = {v: k for k, v in self.categories.items()}

    def __len__(self):
        if self.format == 'coco':
            return len(self.image_ids)
        else:
            return len(self.voc_data)

    def __getitem__(self, idx):
        if self.format == 'coco':
            return self.get_coco_item(idx)
        else:
            return self.get_voc_item(idx)

    def get_coco_item(self, idx):
        """获取COCO格式的数据项"""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # 加载图像
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')

        # 获取标注
        annotations = self.image_annotations.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # COCO bbox格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }

        # 应用变换
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def get_voc_item(self, idx):
        """获取VOC格式的数据项"""
        data = self.voc_data[idx]

        # 加载图像
        img_path = self.images_dir / data['filename']
        image = Image.open(img_path).convert('RGB')

        # 获取标注
        boxes = []
        labels = []

        for obj in data['objects']:
            boxes.append(obj['bbox'])  # [xmin, ymin, xmax, ymax]
            labels.append(self.categories[obj['name']])

        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }

        # 应用变换
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def get_category_names(self):
        """获取类别名称"""
        if self.format == 'coco':
            return list(self.categories.values())
        else:
            return [self.category_names[i] for i in range(1, len(self.categories) + 1)]


class COCODataset(CustomDataset):
    """COCO格式数据集"""

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root, split, transforms, format='coco')


class VOCDataset(CustomDataset):
    """VOC格式数据集"""

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root, split, transforms, format='voc')


def split_coco_dataset(
        train_json_path: str,
        val_json_path: str,
        source_images_dir: str,
        output_dir: str,
        copy_files: bool = True
) -> Dict[str, int]:
    """
    根据COCO标注文件划分训练集和验证集图像

    Args:
        train_json_path (str): 训练集标注文件路径 (train.json)
        val_json_path (str): 验证集标注文件路径 (val.json)
        source_images_dir (str): 原始图像文件夹路径
        output_dir (str): 输出文件夹路径
        copy_files (bool): 是否复制文件，如果为False则移动文件

    Returns:
        Dict[str, int]: 包含处理结果的字典，包括训练集和验证集的图像数量
    """

    # 创建输出目录
    output_path = Path(output_dir)
    train_dir = output_path / "train/images"
    val_dir = output_path / "val/images"

    # 创建目录（如果不存在）
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 读取标注文件
    def load_coco_json(json_path: str) -> List[str]:
        """加载COCO JSON文件并返回图像文件名列表"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取图像文件名
        image_filenames = []
        for image_info in data.get('images', []):
            image_filenames.append(image_info['file_name'])

        return image_filenames

    # 复制或移动文件的函数
    def process_images(image_filenames: List[str], target_dir: Path, dataset_type: str) -> int:
        """处理图像文件（复制或移动）"""
        processed_count = 0
        missing_files = []

        for filename in image_filenames:
            source_path = Path(source_images_dir) / filename
            target_path = target_dir / filename

            if source_path.exists():
                try:
                    if copy_files:
                        shutil.copy2(source_path, target_path)
                    else:
                        shutil.move(str(source_path), str(target_path))
                    processed_count += 1
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
            else:
                missing_files.append(filename)

        # 报告缺失的文件
        if missing_files:
            print(f"警告: {dataset_type}集中有 {len(missing_files)} 个文件未找到:")
            for missing_file in missing_files[:5]:  # 只显示前5个缺失文件
                print(f"  - {missing_file}")
            if len(missing_files) > 5:
                print(f"  ... 还有 {len(missing_files) - 5} 个文件")

        return processed_count

    try:
        # 加载训练集和验证集的图像文件名
        print("正在加载训练集标注文件...")
        train_images = load_coco_json(train_json_path)
        print(f"训练集包含 {len(train_images)} 张图像")

        print("正在加载验证集标注文件...")
        val_images = load_coco_json(val_json_path)
        print(f"验证集包含 {len(val_images)} 张图像")

        # 检查重复图像
        train_set = set(train_images)
        val_set = set(val_images)
        overlap = train_set.intersection(val_set)

        if overlap:
            print(f"警告: 训练集和验证集中有 {len(overlap)} 张重复图像")
            print("重复图像将被复制到两个文件夹中")

        # 处理训练集图像
        print("\n正在处理训练集图像...")
        train_processed = process_images(train_images, train_dir, "训练")
        print(f"训练集: 成功处理 {train_processed}/{len(train_images)} 张图像")

        # 处理验证集图像
        print("\n正在处理验证集图像...")
        val_processed = process_images(val_images, val_dir, "验证")
        print(f"验证集: 成功处理 {val_processed}/{len(val_images)} 张图像")

        # 复制标注文件到对应目录
        print("\n正在复制标注文件...")
        shutil.copy2(train_json_path, train_dir / "annotations.json")
        shutil.copy2(val_json_path, val_dir / "annotations.json")

        # 返回处理结果
        result = {
            "train_images_total": len(train_images),
            "train_images_processed": train_processed,
            "val_images_total": len(val_images),
            "val_images_processed": val_processed,
            "overlap_count": len(overlap)
        }

        print(f"\n数据集划分完成!")
        print(f"训练集文件夹: {train_dir}")
        print(f"验证集文件夹: {val_dir}")

        return result

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        return {}
    except Exception as e:
        print(f"错误: {e}")
        return {}


if __name__ == "__main__":
    # 设置路径
    train_json = "/home/vipuser/amairs/datasets/HRSID_JPG/coco/train/annotations/train.json"  # 训练集标注文件路径
    val_json = "/home/vipuser/amairs/datasets/HRSID_JPG/coco/val/annotations/val.json"  # 验证集标注文件路径
    source_images = "/home/vipuser/amairs/datasets/HRSID_JPG/JPEGImages"  # 原始图像文件夹路径
    output_directory = "/home/vipuser/amairs/datasets/HRSID_JPG/coco"  # 输出目录路径

    # 执行数据集划分
    result = split_coco_dataset(
        train_json_path=train_json,
        val_json_path=val_json,
        source_images_dir=source_images,
        output_dir=output_directory,
        copy_files=True  # True表示复制文件，False表示移动文件
    )

    # 打印结果
    if result:
        print(f"\n处理结果:")
        print(f"训练集: {result['train_images_processed']}/{result['train_images_total']}")
        print(f"验证集: {result['val_images_processed']}/{result['val_images_total']}")
        if result['overlap_count'] > 0:
            print(f"重复图像: {result['overlap_count']}")