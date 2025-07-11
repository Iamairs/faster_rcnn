import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, Any


class DatasetConverter:
    """数据集格式转换器，支持YOLO、COCO、VOC格式相互转换"""

    def __init__(self):
        self.class_names = []
        self.class_to_id = {}
        self.id_to_class = {}

    def set_classes(self, class_names: List[str]):
        """设置类别名称"""
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for idx, name in enumerate(class_names)}

    def get_image_size(self, image_path: str) -> Tuple[int, int]:
        """获取图像尺寸 (width, height)"""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None, None

    # ============ YOLO 相关方法 ============

    def yolo_to_coco(self, yolo_dir: str, output_dir: str, split: str = 'train'):
        """
        YOLO格式转COCO格式
        yolo_dir: YOLO数据集目录，应包含images/和labels/子目录
        output_dir: 输出目录
        split: 数据集分割名称 (train/val/test)
        """
        os.makedirs(output_dir, exist_ok=True)

        images_dir = os.path.join(yolo_dir, split, 'images')
        labels_dir = os.path.join(yolo_dir, split, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Error: {images_dir} or {labels_dir} does not exist")
            return

        # COCO格式数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别信息
        for idx, class_name in enumerate(self.class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })

        annotation_id = 1

        # 处理每个图像
        for image_file in os.listdir(images_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            # 获取图像尺寸
            width, height = self.get_image_size(image_path)
            if width is None or height is None:
                continue

            image_id = len(coco_data["images"]) + 1

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # 处理标注文件
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])

                        # YOLO归一化坐标转换为COCO绝对坐标
                        x_min = (x_center - bbox_width / 2) * width
                        y_min = (y_center - bbox_height / 2) * height
                        bbox_w = bbox_width * width
                        bbox_h = bbox_height * height

                        # 确保边界框在图像范围内
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        bbox_w = min(bbox_w, width - x_min)
                        bbox_h = min(bbox_h, height - y_min)

                        area = bbox_w * bbox_h

                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, bbox_w, bbox_h],
                            "area": area,
                            "iscrowd": 0
                        })

                        annotation_id += 1

        # 保存COCO格式文件
        output_file = os.path.join(output_dir, f'{split}.json')
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"YOLO to COCO conversion completed. Output: {output_file}")

    def yolo_to_voc(self, yolo_dir: str, output_dir: str, split: str = 'train'):
        """
        YOLO格式转VOC格式
        """
        os.makedirs(output_dir, exist_ok=True)

        # 创建VOC目录结构
        annotations_dir = os.path.join(output_dir, 'Annotations')
        images_dir = os.path.join(output_dir, 'JPEGImages')
        imagesets_dir = os.path.join(output_dir, 'ImageSets', 'Main')

        for dir_path in [annotations_dir, images_dir, imagesets_dir]:
            os.makedirs(dir_path, exist_ok=True)

        yolo_images_dir = os.path.join(yolo_dir, 'images', split)
        yolo_labels_dir = os.path.join(yolo_dir, 'labels', split)

        image_names = []

        for image_file in os.listdir(yolo_images_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            image_path = os.path.join(yolo_images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_file)

            # 获取图像尺寸
            width, height = self.get_image_size(image_path)
            if width is None or height is None:
                continue

            # 复制图像文件
            image_name = os.path.splitext(image_file)[0]
            target_image_path = os.path.join(images_dir, f"{image_name}.jpg")
            shutil.copy2(image_path, target_image_path)
            image_names.append(image_name)

            # 创建XML标注文件
            root = ET.Element("annotation")

            # 添加基本信息
            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = f"{image_name}.jpg"

            size_elem = ET.SubElement(root, "size")
            ET.SubElement(size_elem, "width").text = str(width)
            ET.SubElement(size_elem, "height").text = str(height)
            ET.SubElement(size_elem, "depth").text = "3"

            ET.SubElement(root, "segmented").text = "0"

            # 处理标注
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])

                        # YOLO归一化坐标转换为VOC绝对坐标
                        x_min = int((x_center - bbox_width / 2) * width)
                        y_min = int((y_center - bbox_height / 2) * height)
                        x_max = int((x_center + bbox_width / 2) * width)
                        y_max = int((y_center + bbox_height / 2) * height)

                        # 确保坐标在图像范围内
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(width, x_max)
                        y_max = min(height, y_max)

                        class_name = self.id_to_class.get(class_id, f"class_{class_id}")

                        obj_elem = ET.SubElement(root, "object")
                        ET.SubElement(obj_elem, "name").text = class_name
                        ET.SubElement(obj_elem, "pose").text = "Unspecified"
                        ET.SubElement(obj_elem, "truncated").text = "0"
                        ET.SubElement(obj_elem, "difficult").text = "0"

                        bndbox_elem = ET.SubElement(obj_elem, "bndbox")
                        ET.SubElement(bndbox_elem, "xmin").text = str(x_min)
                        ET.SubElement(bndbox_elem, "ymin").text = str(y_min)
                        ET.SubElement(bndbox_elem, "xmax").text = str(x_max)
                        ET.SubElement(bndbox_elem, "ymax").text = str(y_max)

            # 保存XML文件
            tree = ET.ElementTree(root)
            xml_path = os.path.join(annotations_dir, f"{image_name}.xml")
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)

        # 创建ImageSets文件
        with open(os.path.join(imagesets_dir, f'{split}.txt'), 'w') as f:
            for name in image_names:
                f.write(f"{name}\n")

        print(f"YOLO to VOC conversion completed. Output: {output_dir}")

    # ============ COCO 相关方法 ============

    def coco_to_yolo(self, coco_file: str, output_dir: str, split: str = 'train'):
        """
        COCO格式转YOLO格式
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # 创建输出目录
        images_output_dir = os.path.join(output_dir, split, 'images')
        labels_output_dir = os.path.join(output_dir, split, 'labels')

        for dir_path in [images_output_dir, labels_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 构建图像ID到信息的映射
        image_info = {img['id']: img for img in coco_data['images']}

        # 构建类别ID映射
        category_mapping = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

        # 按图像分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        # 处理每个图像
        for image_id, image_data in image_info.items():
            file_name = image_data['file_name']
            width = image_data['width']
            height = image_data['height']

            # 创建YOLO标注文件
            label_file = os.path.splitext(file_name)[0] + '.txt'
            label_path = os.path.join(labels_output_dir, label_file)

            with open(label_path, 'w') as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        bbox = ann['bbox']  # [x, y, width, height]
                        x, y, w, h = bbox

                        # COCO绝对坐标转换为YOLO归一化坐标
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        norm_width = w / width
                        norm_height = h / height

                        # 确保坐标在[0,1]范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))

                        class_id = category_mapping.get(ann['category_id'], 0)

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

        print(f"COCO to YOLO conversion completed. Output: {output_dir}")

    def coco_to_voc(self, coco_file: str, images_dir: str, output_dir: str):
        """
        COCO格式转VOC格式
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # 创建VOC目录结构
        annotations_dir = os.path.join(output_dir, 'Annotations')
        voc_images_dir = os.path.join(output_dir, 'JPEGImages')
        imagesets_dir = os.path.join(output_dir, 'ImageSets', 'Main')

        for dir_path in [annotations_dir, voc_images_dir, imagesets_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 构建类别映射
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # 按图像分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        image_names = []

        # 处理每个图像
        for image_data in coco_data['images']:
            file_name = image_data['file_name']
            image_id = image_data['id']
            width = image_data['width']
            height = image_data['height']

            # 复制图像文件
            source_path = os.path.join(images_dir, file_name)
            if not os.path.exists(source_path):
                print(f"Warning: Image {source_path} not found")
                continue

            image_name = os.path.splitext(file_name)[0]
            target_path = os.path.join(voc_images_dir, f"{image_name}.jpg")
            shutil.copy2(source_path, target_path)
            image_names.append(image_name)

            # 创建XML标注文件
            root = ET.Element("annotation")

            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = f"{image_name}.jpg"

            size_elem = ET.SubElement(root, "size")
            ET.SubElement(size_elem, "width").text = str(width)
            ET.SubElement(size_elem, "height").text = str(height)
            ET.SubElement(size_elem, "depth").text = "3"

            ET.SubElement(root, "segmented").text = "0"

            # 添加标注
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox

                    x_min = int(x)
                    y_min = int(y)
                    x_max = int(x + w)
                    y_max = int(y + h)

                    # 确保坐标在图像范围内
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)

                    class_name = categories.get(ann['category_id'], 'unknown')

                    obj_elem = ET.SubElement(root, "object")
                    ET.SubElement(obj_elem, "name").text = class_name
                    ET.SubElement(obj_elem, "pose").text = "Unspecified"
                    ET.SubElement(obj_elem, "truncated").text = "0"
                    ET.SubElement(obj_elem, "difficult").text = "0"

                    bndbox_elem = ET.SubElement(obj_elem, "bndbox")
                    ET.SubElement(bndbox_elem, "xmin").text = str(x_min)
                    ET.SubElement(bndbox_elem, "ymin").text = str(y_min)
                    ET.SubElement(bndbox_elem, "xmax").text = str(x_max)
                    ET.SubElement(bndbox_elem, "ymax").text = str(y_max)

            # 保存XML文件
            tree = ET.ElementTree(root)
            xml_path = os.path.join(annotations_dir, f"{image_name}.xml")
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)

        # 创建ImageSets文件
        with open(os.path.join(imagesets_dir, 'trainval.txt'), 'w') as f:
            for name in image_names:
                f.write(f"{name}\n")

        print(f"COCO to VOC conversion completed. Output: {output_dir}")

    # ============ VOC 相关方法 ============

    def voc_to_yolo(self, voc_dir: str, output_dir: str, split: str = 'trainval'):
        """
        VOC格式转YOLO格式
        """
        # 创建输出目录
        images_output_dir = os.path.join(output_dir, 'images', 'train')
        labels_output_dir = os.path.join(output_dir, 'labels', 'train')

        for dir_path in [images_output_dir, labels_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        annotations_dir = os.path.join(voc_dir, 'Annotations')
        images_dir = os.path.join(voc_dir, 'JPEGImages')
        imagesets_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{split}.txt')

        # 读取图像列表
        if os.path.exists(imagesets_file):
            with open(imagesets_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
        else:
            # 如果没有ImageSets文件，使用所有图像
            image_names = [os.path.splitext(f)[0] for f in os.listdir(images_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for image_name in image_names:
            # 查找图像文件
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                potential_path = os.path.join(images_dir, image_name + ext)
                if os.path.exists(potential_path):
                    image_file = image_name + ext
                    break

            if image_file is None:
                print(f"Warning: Image file for {image_name} not found")
                continue

            image_path = os.path.join(images_dir, image_file)
            xml_path = os.path.join(annotations_dir, f"{image_name}.xml")

            if not os.path.exists(xml_path):
                print(f"Warning: Annotation file {xml_path} not found")
                continue

            # 获取图像尺寸
            width, height = self.get_image_size(image_path)
            if width is None or height is None:
                continue

            # 复制图像文件
            target_image_path = os.path.join(images_output_dir, image_file)
            shutil.copy2(image_path, target_image_path)

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 创建YOLO标注文件
            label_file = f"{image_name}.txt"
            label_path = os.path.join(labels_output_dir, label_file)

            with open(label_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in self.class_to_id:
                        print(f"Warning: Unknown class {class_name}")
                        continue

                    class_id = self.class_to_id[class_name]

                    bndbox = obj.find('bndbox')
                    x_min = float(bndbox.find('xmin').text)
                    y_min = float(bndbox.find('ymin').text)
                    x_max = float(bndbox.find('xmax').text)
                    y_max = float(bndbox.find('ymax').text)

                    # VOC绝对坐标转换为YOLO归一化坐标
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height

                    # 确保坐标在[0,1]范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    bbox_width = max(0, min(1, bbox_width))
                    bbox_height = max(0, min(1, bbox_height))

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        print(f"VOC to YOLO conversion completed. Output: {output_dir}")

    def voc_to_coco(self, voc_dir: str, output_file: str, split: str = 'trainval'):
        """
        VOC格式转COCO格式
        """
        annotations_dir = os.path.join(voc_dir, 'Annotations')
        images_dir = os.path.join(voc_dir, 'JPEGImages')
        imagesets_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{split}.txt')

        # 读取图像列表
        if os.path.exists(imagesets_file):
            with open(imagesets_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
        else:
            image_names = [os.path.splitext(f)[0] for f in os.listdir(images_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # COCO格式数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别信息
        for idx, class_name in enumerate(self.class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })

        annotation_id = 1

        for image_id, image_name in enumerate(image_names, 1):
            # 查找图像文件
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                potential_path = os.path.join(images_dir, image_name + ext)
                if os.path.exists(potential_path):
                    image_file = image_name + ext
                    break

            if image_file is None:
                continue

            image_path = os.path.join(images_dir, image_file)
            xml_path = os.path.join(annotations_dir, f"{image_name}.xml")

            if not os.path.exists(xml_path):
                continue

            # 获取图像尺寸
            width, height = self.get_image_size(image_path)
            if width is None or height is None:
                continue

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in self.class_to_id:
                    continue

                class_id = self.class_to_id[class_name]

                bndbox = obj.find('bndbox')
                x_min = float(bndbox.find('xmin').text)
                y_min = float(bndbox.find('ymin').text)
                x_max = float(bndbox.find('xmax').text)
                y_max = float(bndbox.find('ymax').text)

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                area = bbox_width * bbox_height

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0
                })

                annotation_id += 1

        # 保存COCO格式文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"VOC to COCO conversion completed. Output: {output_file}")


def main():
    converter = DatasetConverter()

    # 设置类别名称（根据你的数据集修改）
    class_names = ['ship']
    converter.set_classes(class_names)

    # 示例：YOLO转COCO
    # converter.yolo_to_coco(
    #     yolo_dir=r'E:\edward\detection\yolo_v10\datasets\SSDD',
    #     output_dir=r'E:\edward\detection\faster_rcnn\datasets\SSDD',
    #     split='val'
    # )

    # 示例：YOLO转VOC
    # converter.yolo_to_voc(
    #     yolo_dir='path/to/yolo/dataset',
    #     output_dir='path/to/output/voc',
    #     split='train'
    # )

    # 示例：COCO转YOLO
    converter.coco_to_yolo(
        coco_file='/home/vipuser/amairs/datasets/HRSID_JPG/annotations/test2017.json',
        output_dir='/home/vipuser/amairs/datasets/HRSID_JPG/yolo',
        split='val'
    )

    # 示例：COCO转VOC
    # converter.coco_to_voc(
    #     coco_file='path/to/coco/annotations.json',
    #     images_dir='path/to/coco/images',
    #     output_dir='path/to/output/voc'
    # )

    # 示例：VOC转YOLO
    # converter.voc_to_yolo(
    #     voc_dir='path/to/voc/dataset',
    #     output_dir='path/to/output/yolo',
    #     split='trainval'
    # )

    # 示例：VOC转COCO
    # converter.voc_to_coco(
    #     voc_dir='path/to/voc/dataset',
    #     output_file='path/to/output/coco/annotations.json',
    #     split='trainval'
    # )

    print("转换完成！")


if __name__ == "__main__":
    main()