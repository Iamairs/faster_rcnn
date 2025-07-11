"""
数据变换和增强
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from PIL import Image
import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if target is not None and "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            if target is not None and "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
                target["boxes"] = bbox
        return image, target


class RandomRotation:
    def __init__(self, degrees=(-10, 10)):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, image, target):
        if random.random() < 0.5:  # 50%概率进行旋转
            angle = random.uniform(self.degrees[0], self.degrees[1])

            # 将tensor转换为PIL Image进行旋转
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)

            # 旋转图像
            image = F.rotate(image, angle, expand=False)

            # 转换回tensor
            image = F.to_tensor(image)

            # 边框的旋转变换比较复杂，暂时跳过
            
        return image, target


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        if any([brightness, contrast, saturation, hue]):
            self.color_jitter = T.ColorJitter(
                brightness=brightness,
                contrast=contrast, 
                saturation=saturation,
                hue=hue
            )
        else:
            self.color_jitter = None
    
    def __call__(self, image, target):
        if self.color_jitter is not None:
            # 将tensor转换为PIL Image
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)
            
            image = self.color_jitter(image)
            
            # 转换回tensor
            image = F.to_tensor(image)
        
        return image, target


class RandomResize:
    def __init__(self, min_size=800, max_size=1333, keep_ratio=True):
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio
    
    def __call__(self, image, target):
        original_height, original_width = image.shape[-2:]
        
        if self.keep_ratio:
            # 保持长宽比的随机缩放
            scale_factor = random.uniform(
                self.min_size / min(original_height, original_width),
                self.max_size / max(original_height, original_width)
            )
            
            new_height = int(original_height * scale_factor)
            new_width = int(original_width * scale_factor)
            
            # 确保不超过最大尺寸
            if max(new_height, new_width) > self.max_size:
                if new_height > new_width:
                    new_height = self.max_size
                    new_width = int(original_width * (self.max_size / original_height))
                else:
                    new_width = self.max_size
                    new_height = int(original_height * (self.max_size / original_width))
        else:
            # 随机选择目标尺寸
            new_height = random.randint(self.min_size, self.max_size)
            new_width = random.randint(self.min_size, self.max_size)
        
        # 调整图像尺寸
        image = F.resize(image, (new_height, new_width))
        
        # 调整边框坐标
        if target is not None and "boxes" in target:
            height_ratio = new_height / original_height
            width_ratio = new_width / original_width
            
            target["boxes"][:, 0] *= width_ratio   # x1
            target["boxes"][:, 1] *= height_ratio  # y1
            target["boxes"][:, 2] *= width_ratio   # x2
            target["boxes"][:, 3] *= height_ratio  # y2
            
            if "area" in target:
                target["area"] *= (height_ratio * width_ratio)
        
        return image, target


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, image, target):
        original_height, original_width = image.shape[-2:]
        new_height, new_width = self.size
        
        # 调整图像尺寸
        image = F.resize(image, self.size)

        if target is not None and "boxes" in target:
            boxes = target["boxes"]

            # 1. 处理没有物体的情况 (例如，背景图片)
            # 你的数据集中也可能存在这种情况
            if boxes.numel() == 0:
                return image, target

            # 2. 检查维度
            if boxes.dim() == 1:
                # 如果是 (4,) 的一维张量, 增加一个维度变为 (1, 4)
                boxes = boxes.unsqueeze(0)

            height_ratio = new_height / original_height
            width_ratio = new_width / original_width

            boxes[:, 0] *= width_ratio  # x1
            boxes[:, 1] *= height_ratio  # y1
            boxes[:, 2] *= width_ratio  # x2
            boxes[:, 3] *= height_ratio  # y2
            
            if "area" in target:
                target["area"] *= (height_ratio * width_ratio)
        
        return image, target


class RandomCrop:
    def __init__(self, size, padding=None):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, image, target):
        if self.padding is not None:
            image = F.pad(image, self.padding)
        
        # 获取随机裁剪参数
        i, j, h, w = T.RandomCrop.get_params(image, self.size)
        
        # 裁剪图像
        image = F.crop(image, i, j, h, w)
        
        # 调整边框坐标
        if target is not None and "boxes" in target:
            boxes = target["boxes"]
            
            # 调整坐标
            boxes[:, 0] -= j  # x1
            boxes[:, 1] -= i  # y1
            boxes[:, 2] -= j  # x2
            boxes[:, 3] -= i  # y2
            
            # 裁剪边框
            boxes[:, 0].clamp_(min=0, max=w)
            boxes[:, 1].clamp_(min=0, max=h)
            boxes[:, 2].clamp_(min=0, max=w)
            boxes[:, 3].clamp_(min=0, max=h)
            
            # 过滤掉无效的边框
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            for key in target.keys():
                if isinstance(target[key], torch.Tensor) and len(target[key]) == len(boxes):
                    target[key] = target[key][valid_boxes]
        
        return image, target


def get_transforms(train=True, config=None):
    """根据配置文件获取数据变换"""
    transforms = []
    
    # 转换为张量
    transforms.append(ToTensor())
    
    if train and config is not None:
        # 训练时的数据增强
        aug_config = config.get('training', {}).get('augmentation', {})
        
        # 水平翻转
        if aug_config.get('horizontal_flip', 0) > 0:
            transforms.append(RandomHorizontalFlip(aug_config['horizontal_flip']))
        
        # 垂直翻转
        if aug_config.get('vertical_flip', 0) > 0:
            transforms.append(RandomVerticalFlip(aug_config['vertical_flip']))
        
        # # 旋转增强（跳过）
        # if aug_config.get('random_rotation', 0) > 0:
        #     degrees = aug_config['random_rotation']
        #     transforms.append(RandomRotation(degrees))
        
        # 颜色抖动
        color_jitter = aug_config.get('color_jitter', {})
        if any(color_jitter.values()):
            transforms.append(ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0)
            ))
        
        # 随机缩放
        random_resize = aug_config.get('random_resize', {})
        if random_resize:
            transforms.append(RandomResize(
                min_size=random_resize.get('min_size', 640),
                max_size=random_resize.get('max_size', 960)
            ))
        
    else:
        # 测试时只调整尺寸
        if config is not None:
            eval_image_size = config.get('validation', {}).get('eval_image_size', 800)
            random_resize = config.get('training', {}).get('augmentation', {}).get('random_resize', {})
            transforms.append(Resize(eval_image_size))
        else:
            transforms.append(Resize(800))
    
    # 标准化
    transforms.append(Normalize())
    
    return Compose(transforms)


def get_basic_transforms(train=True):
    """获取基础数据变换（不依赖配置文件）"""
    transforms = []
    
    # 转换为张量
    transforms.append(ToTensor())
    
    if train:
        # 基础训练增强
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomResize(min_size=800, max_size=1333))
    else:
        # 测试时固定尺寸
        transforms.append(Resize(800))
    
    # 标准化
    transforms.append(Normalize())
    
    return Compose(transforms)


# 测试函数
def test_transforms():
    """测试数据变换功能"""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # 创建测试图像和目标
    image = Image.new('RGB', (640, 480), color='red')
    target = {
        'boxes': torch.tensor([[100, 100, 200, 200], [300, 200, 400, 300]], dtype=torch.float32),
        'labels': torch.tensor([1, 2], dtype=torch.int64),
        'area': torch.tensor([10000, 10000], dtype=torch.float32),
        'image_id': torch.tensor([1], dtype=torch.int64),
        'iscrowd': torch.tensor([0, 0], dtype=torch.int64)
    }
    
    # 测试配置
    test_config = {
        'training': {
            'augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'random_rotation': 15,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1
                },
                'random_resize': {
                    'min_size': 600,
                    'max_size': 1000
                }
            }
        }
    }
    
    # 获取变换
    train_transforms = get_transforms(train=True, config=test_config)
    val_transforms = get_transforms(train=False, config=test_config)
    
    # 应用变换
    train_img, train_target = train_transforms(image, target)
    val_img, val_target = val_transforms(image, target)
    
    print("训练变换结果:")
    print(f"图像尺寸: {train_img.shape}")
    print(f"边框: {train_target['boxes']}")
    
    print("\n验证变换结果:")
    print(f"图像尺寸: {val_img.shape}")
    print(f"边框: {val_target['boxes']}")


if __name__ == "__main__":
    test_transforms()