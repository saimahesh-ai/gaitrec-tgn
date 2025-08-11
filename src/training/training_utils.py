import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
import logging
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    # Model parameters
    input_channels: int = 10
    embedding_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    num_classes: int = 5
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    
    # Loss weights
    classification_weight: float = 0.4
    progression_weight: float = 0.3
    asymmetry_weight: float = 0.1
    temporal_consistency_weight: float = 0.1
    pathology_specific_weight: float = 0.1
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'plateau'  # 'plateau', 'cosine', 'step'
    warmup_epochs: int = 5
    
    # Data
    data_path: str = 'data'
    cache_dir: str = 'cache'
    save_dir: str = 'checkpoints'
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 5
    
    # Advanced
    use_mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metrics, 'max' for maximizing
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        logger.info(f"Initialized EarlyStopping with patience={patience}, mode={mode}")
    
    def __call__(self, metric: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta
        
        if improved:
            self.best_score = metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        
        return False
    
    def get_best_score(self) -> float:
        """Get the best score achieved."""
        return self.best_score if self.best_score is not None else float('inf')


class ModelCheckpoint:
    """
    Model checkpointing utility.
    """
    
    def __init__(self, save_dir: str, save_top_k: int = 3, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        Initialize model checkpointing.
        
        Args:
            save_dir: Directory to save checkpoints
            save_top_k: Number of best models to keep
            monitor: Metric to monitor for best models
            mode: 'min' for minimizing metrics, 'max' for maximizing
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        
        self.best_models = []
        self.best_scores = []
        
        logger.info(f"Initialized ModelCheckpoint: save_dir={save_dir}, "
                   f"save_top_k={save_top_k}, monitor={monitor}")
    
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, float],
                       optimizer=None, scheduler=None, config=None) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            config: Training configuration
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Update best models list
        self._update_best_models(checkpoint_path, metrics[self.monitor])
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _update_best_models(self, checkpoint_path: Path, score: float):
        """Update list of best models."""
        if len(self.best_models) < self.save_top_k:
            self.best_models.append(checkpoint_path)
            self.best_scores.append(score)
        else:
            # Find worst model to replace
            if self.mode == 'min':
                worst_idx = np.argmax(self.best_scores)
                if score < self.best_scores[worst_idx]:
                    # Remove old checkpoint
                    old_checkpoint = self.best_models[worst_idx]
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                    
                    # Update lists
                    self.best_models[worst_idx] = checkpoint_path
                    self.best_scores[worst_idx] = score
            else:
                worst_idx = np.argmin(self.best_scores)
                if score > self.best_scores[worst_idx]:
                    # Remove old checkpoint
                    old_checkpoint = self.best_models[worst_idx]
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                    
                    # Update lists
                    self.best_models[worst_idx] = checkpoint_path
                    self.best_scores[worst_idx] = score
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints that are no longer in top-k."""
        all_checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        
        for checkpoint in all_checkpoints:
            if checkpoint not in self.best_models:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if not self.best_models:
            return None
        
        if self.mode == 'min':
            best_idx = np.argmin(self.best_scores)
        else:
            best_idx = np.argmax(self.best_scores)
        
        return self.best_models[best_idx]
    
    def load_best_checkpoint(self, model: nn.Module, optimizer=None, scheduler=None) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Checkpoint data
        """
        best_checkpoint = self.get_best_checkpoint()
        if best_checkpoint is None:
            raise ValueError("No checkpoints available")
        
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded best checkpoint from {best_checkpoint}")
        return checkpoint


class TrainingMonitor:
    """
    Monitor training progress and generate reports.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize training monitor.
        
        Args:
            save_dir: Directory to save monitoring data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }
        
        logger.info(f"Initialized TrainingMonitor: save_dir={save_dir}")
    
    def update(self, epoch: int, train_metrics: Dict[str, float], 
               val_metrics: Dict[str, float], learning_rate: float):
        """
        Update monitoring data.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
        """
        # Update training metrics
        for key in ['loss', 'accuracy', 'f1']:
            if key in train_metrics:
                self.metrics_history[f'train_{key}'].append(train_metrics[key])
        
        # Update validation metrics
        for key in ['loss', 'accuracy', 'f1']:
            if key in val_metrics:
                self.metrics_history[f'val_{key}'].append(val_metrics[key])
        
        # Update learning rate
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Save to file
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.save_dir / "training_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[f'{metric_name}_final'] = values[-1]
                summary[f'{metric_name}_best'] = min(values) if 'loss' in metric_name else max(values)
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            if self.metrics_history['train_loss']:
                axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
                axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
                axes[0, 0].set_title('Training and Validation Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Accuracy curves
            if self.metrics_history['train_acc']:
                axes[0, 1].plot(self.metrics_history['train_acc'], label='Train Acc')
                axes[0, 1].plot(self.metrics_history['val_acc'], label='Val Acc')
                axes[0, 1].set_title('Training and Validation Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # F1 curves
            if self.metrics_history['train_f1']:
                axes[1, 0].plot(self.metrics_history['train_f1'], label='Train F1')
                axes[1, 0].plot(self.metrics_history['val_f1'], label='Val F1')
                axes[1, 0].set_title('Training and Validation F1 Score')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate curve
            if self.metrics_history['learning_rate']:
                axes[1, 1].plot(self.metrics_history['learning_rate'])
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
