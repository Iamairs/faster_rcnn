"""
日志记录工具
"""
import logging
import os
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger('training', self.log_dir / 'training.log')

        # 训练记录
        self.train_losses = []
        self.val_maps = []
        self.learning_rates = []

    def log_epoch(self, epoch, train_loss, val_map=None, lr=None):
        """记录一个epoch的结果"""
        self.train_losses.append(train_loss)

        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"

        if val_map is not None:
            self.val_maps.append(val_map)
            log_msg += f", Val mAP = {val_map:.4f}"

        if lr is not None:
            self.learning_rates.append(lr)
            log_msg += f", LR = {lr:.6f}"

        self.logger.info(log_msg)

    def save_curves(self):
        """保存训练曲线"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 训练损失
        axes[0].plot(self.train_losses)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # 验证mAP
        if self.val_maps:
            axes[1].plot(self.val_maps)
            axes[1].set_title('Validation mAP')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('mAP')
            axes[1].grid(True)

        # 学习率
        if self.learning_rates:
            axes[2].plot(self.learning_rates)
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('LR')
            axes[2].set_yscale('log')
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300)
        plt.close()