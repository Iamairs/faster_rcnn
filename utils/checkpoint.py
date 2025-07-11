"""
模型检查点工具
"""
import torch
from pathlib import Path
import shutil


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """保存检查点"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_filepath = save_dir / 'best_model.pth'
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path):
    """加载检查点"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


class ModelSaver:
    """模型保存器"""

    def __init__(self, save_dir, save_freq=10, max_keep=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.max_keep = max_keep
        self.best_score = 0.0
        self.saved_models = []

    def save(self, model, optimizer, scheduler, epoch, score, is_best=False):
        """保存模型"""
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'score': score
        }

        # 保存最新模型
        latest_path = self.save_dir / 'latest.pth'
        torch.save(state, latest_path)

        # 保存最佳模型
        if is_best or score > self.best_score:
            self.best_score = score
            best_path = self.save_dir / 'best.pth'
            torch.save(state, best_path)

        # 定期保存
        if epoch % self.save_freq == 0:
            epoch_path = self.save_dir / f'epoch_{epoch}.pth'
            torch.save(state, epoch_path)
            self.saved_models.append(epoch_path)

            # 清理旧模型
            self._cleanup_old_models()

    def _cleanup_old_models(self):
        """清理旧模型文件"""
        if len(self.saved_models) > self.max_keep:
            old_models = self.saved_models[:-self.max_keep]
            for model_path in old_models:
                if model_path.exists():
                    model_path.unlink()
            self.saved_models = self.saved_models[-self.max_keep:]