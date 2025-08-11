import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from tqdm import tqdm
import time
from pathlib import Path

from ..models import GaitProgressionLoss, CombinedGaitLoss
from ..preprocessing import GaitDataPreprocessor

logger = logging.getLogger(__name__)

class GaitTGNTrainer:
    """
    Main trainer class for the GaitRec Temporal Graph Network system.
    Handles training, validation, and testing of the complete pipeline.
    """
    
    def __init__(self, state_encoder: nn.Module, tgn_model: nn.Module, 
                 device: str = 'auto', config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            state_encoder: Gait state encoder model
            tgn_model: Temporal Graph Network model
            device: Device to use ('auto', 'cuda', 'cpu')
            config: Training configuration dictionary
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Models
        self.state_encoder = state_encoder.to(self.device)
        self.tgn_model = tgn_model.to(self.device)
        
        # Configuration
        self.config = config or {}
        self._setup_default_config()
        
        # Loss functions
        self.loss_fn = CombinedGaitLoss(self.config.get('loss_config', {}))
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }
        
        logger.info(f"Initialized GaitTGNTrainer on device: {self.device}")
        logger.info(f"State encoder parameters: {sum(p.numel() for p in self.state_encoder.parameters()):,}")
        logger.info(f"TGN model parameters: {sum(p.numel() for p in self.tgn_model.parameters()):,}")
    
    def _setup_default_config(self):
        """Setup default training configuration."""
        defaults = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'num_epochs': 100,
            'patience': 10,
            'gradient_clip': 1.0,
            'save_dir': 'checkpoints',
            'log_interval': 10,
            'eval_interval': 1,
            'use_mixed_precision': False
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup the optimizer."""
        # Combine parameters from both models
        all_params = list(self.state_encoder.parameters()) + list(self.tgn_model.parameters())
        
        if self.config.get('optimizer', 'adam').lower() == 'adamw':
            optimizer = optim.AdamW(
                all_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                all_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup the learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['num_epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.state_encoder.train()
        self.tgn_model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass
            loss, metrics = self._training_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.state_encoder.parameters()) + list(self.tgn_model.parameters()),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = batch.get('batch_size', len(batch.get('x', [])))
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if 'accuracy' in metrics:
                correct_predictions += metrics['accuracy'] * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{metrics.get('accuracy', 0):.3f}"
            })
            
            # Log at intervals
            if batch_idx % self.config['log_interval'] == 0:
                logger.debug(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                           f"Acc={metrics.get('accuracy', 0):.3f}")
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0
        }
        
        return epoch_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract data
        x = batch['x']
        edge_index = batch['edge_index']
        labels = batch['labels']
        timestamps = batch.get('timestamps')
        batch_idx = batch.get('batch')
        
        # Get embeddings from state encoder
        embeddings = self.state_encoder(x)
        
        # Forward pass through TGN
        class_pred, prog_pred = self.tgn_model(
            embeddings, edge_index, timestamps, batch_idx
        )
        
        # Prepare predictions and targets for loss computation
        predictions = {
            'classification': class_pred,
            'progression': prog_pred
        }
        
        targets = {
            'labels': labels
        }
        
        # Add next state if available for progression loss
        if 'next_state' in batch:
            targets['next_state'] = batch['next_state']
        
        # Compute loss
        loss, loss_components = self.loss_fn(predictions, targets)
        
        # Compute accuracy
        pred_labels = torch.argmax(class_pred, dim=1)
        accuracy = (pred_labels == labels).float().mean().item()
        
        metrics = {
            'accuracy': accuracy,
            **loss_components
        }
        
        return loss, metrics
    
    def _prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training.
        
        Args:
            batch: Raw batch from data loader
            
        Returns:
            Prepared batch dictionary
        """
        if hasattr(batch, 'to'):
            # PyTorch Geometric Data object
            batch = batch.to(self.device)
            return {
                'x': batch.x,
                'edge_index': batch.edge_index,
                'labels': batch.y,
                'timestamps': getattr(batch, 'timestamps', None),
                'batch': getattr(batch, 'batch', None),
                'batch_size': batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.x)
            }
        else:
            # Custom batch format
            return {k: v.to(self.device) if hasattr(v, 'to') else v 
                   for k, v in batch.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.state_encoder.eval()
        self.tgn_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._prepare_batch(batch)
                
                # Forward pass
                loss, metrics = self._validation_step(batch)
                
                # Update metrics
                batch_size = batch.get('batch_size', len(batch.get('x', [])))
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions for detailed metrics
                if 'predictions' in metrics:
                    all_predictions.extend(metrics['predictions'].cpu().numpy())
                    all_labels.extend(metrics['labels'].cpu().numpy())
        
        # Compute validation metrics
        val_metrics = {
            'loss': total_loss / total_samples,
            'accuracy': metrics.get('accuracy', 0.0)
        }
        
        # Compute additional metrics if predictions available
        if all_predictions and all_labels:
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            val_metrics.update({
                'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
                'f1_weighted': f1_score(all_labels, all_predictions, average='weighted'),
                'precision_macro': precision_score(all_labels, all_predictions, average='macro'),
                'recall_macro': recall_score(all_labels, all_predictions, average='macro')
            })
        
        return val_metrics
    
    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single validation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract data
        x = batch['x']
        edge_index = batch['edge_index']
        labels = batch['labels']
        timestamps = batch.get('timestamps')
        batch_idx = batch.get('batch')
        
        # Get embeddings from state encoder
        embeddings = self.state_encoder(x)
        
        # Forward pass through TGN
        class_pred, prog_pred = self.tgn_model(
            embeddings, edge_index, timestamps, batch_idx
        )
        
        # Prepare predictions and targets for loss computation
        predictions = {
            'classification': class_pred,
            'progression': prog_pred
        }
        
        targets = {
            'labels': labels
        }
        
        if 'next_state' in batch:
            targets['next_state'] = batch['next_state']
        
        # Compute loss
        loss, loss_components = self.loss_fn(predictions, targets)
        
        # Compute accuracy
        pred_labels = torch.argmax(class_pred, dim=1)
        accuracy = (pred_labels == labels).float().mean().item()
        
        metrics = {
            'accuracy': accuracy,
            'predictions': pred_labels,
            'labels': labels,
            **loss_components
        }
        
        return loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        
        # Create checkpoint directory
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if epoch % self.config['eval_interval'] == 0:
                val_metrics = self.validate(val_loader)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Log metrics
                logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
                logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"Acc: {train_metrics['accuracy']:.4f}")
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                           f"Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_metric:
                    self.best_val_metric = val_metrics['loss']
                    self.save_checkpoint(save_dir / 'best_model.pth', val_metrics)
                
                # Update training history
                self._update_history(train_metrics, val_metrics)
            else:
                # Only update training metrics
                self._update_history(train_metrics, {})
        
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pth', {})
        
        logger.info("Training completed!")
        return self.training_history
    
    def _update_history(self, train_metrics: Dict[str, float], 
                       val_metrics: Dict[str, float]):
        """Update training history."""
        self.training_history['train_loss'].append(train_metrics.get('loss', 0.0))
        self.training_history['train_acc'].append(train_metrics.get('accuracy', 0.0))
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics.get('loss', 0.0))
            self.training_history['val_acc'].append(val_metrics.get('accuracy', 0.0))
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['learning_rate'].append(current_lr)
    
    def save_checkpoint(self, filepath: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'tgn_model_state_dict': self.tgn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.tgn_model.load_state_dict(checkpoint['tgn_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Checkpoint loaded from {filepath} (epoch {self.current_epoch})")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing test metrics
        """
        logger.info("Starting evaluation...")
        
        # Load best model if available
        best_model_path = Path(self.config['save_dir']) / 'best_model.pth'
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
            logger.info("Loaded best model for evaluation")
        
        return self.validate(test_loader)

