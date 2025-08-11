import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GaitProgressionLoss(nn.Module):
    """
    Combined loss for classification and progression prediction.
    Balances multiple objectives: classification accuracy, progression prediction,
    and temporal consistency.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Initialize the combined loss function.
        
        Args:
            alpha: Weight for classification loss
            beta: Weight for progression loss
            gamma: Weight for consistency loss
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Individual loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Validation
        total_weight = alpha + beta + gamma
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {total_weight}, normalizing to 1.0")
            self.alpha /= total_weight
            self.beta /= total_weight
            self.gamma /= total_weight
        
        logger.info(f"Initialized GaitProgressionLoss with weights: "
                   f"α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
    
    def forward(self, class_pred: torch.Tensor, prog_pred: torch.Tensor, 
                class_true: torch.Tensor, next_state: Optional[torch.Tensor] = None, 
                consistency_pairs: Optional[list] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss.
        
        Args:
            class_pred: Classification predictions
            prog_pred: Progression predictions
            class_true: True classification labels
            next_state: Optional next state for progression loss
            consistency_pairs: Optional pairs of indices for consistency loss
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Classification loss
        loss_class = self.ce_loss(class_pred, class_true)
        
        # Progression loss (if next state available)
        loss_prog = torch.tensor(0.0, device=class_pred.device)
        if next_state is not None:
            loss_prog = self.mse_loss(prog_pred, next_state)
        
        # Consistency loss (similar states should have similar predictions)
        loss_consist = torch.tensor(0.0, device=class_pred.device)
        if consistency_pairs is not None and len(consistency_pairs) > 0:
            for i, j in consistency_pairs:
                if i < len(prog_pred) and j < len(prog_pred):
                    loss_consist += self.mse_loss(prog_pred[i], prog_pred[j])
            loss_consist /= len(consistency_pairs)
        
        # Combined loss
        total_loss = (self.alpha * loss_class + 
                     self.beta * loss_prog + 
                     self.gamma * loss_consist)
        
        # Loss components for monitoring
        loss_components = {
            'classification': loss_class.item(),
            'progression': loss_prog.item() if isinstance(loss_prog, torch.Tensor) else 0.0,
            'consistency': loss_consist.item() if isinstance(loss_consist, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


class AsymmetryLoss(nn.Module):
    """
    Loss function to encourage symmetric gait patterns between left and right limbs.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize the asymmetry loss.
        
        Args:
            weight: Weight for the asymmetry penalty
        """
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        
        logger.info(f"Initialized AsymmetryLoss with weight {weight}")
    
    def forward(self, left_features: torch.Tensor, right_features: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetry loss between left and right limb features.
        
        Args:
            left_features: Features from left limb
            right_features: Features from right limb
            
        Returns:
            Asymmetry loss value
        """
        # Ensure same shape
        if left_features.shape != right_features.shape:
            # Pad or truncate to match
            min_len = min(left_features.shape[0], right_features.shape[0])
            left_features = left_features[:min_len]
            right_features = right_features[:min_len]
        
        # Compute asymmetry loss
        asymmetry_loss = self.mse_loss(left_features, right_features)
        
        return self.weight * asymmetry_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Loss function to ensure temporal consistency in gait progression.
    """
    
    def __init__(self, weight: float = 0.2, window_size: int = 3):
        """
        Initialize the temporal consistency loss.
        
        Args:
            weight: Weight for the consistency penalty
            window_size: Size of the temporal window for consistency checking
        """
        super().__init__()
        self.weight = weight
        self.window_size = window_size
        self.mse_loss = nn.MSELoss()
        
        logger.info(f"Initialized TemporalConsistencyLoss with weight {weight}, "
                   f"window_size {window_size}")
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            sequence: Temporal sequence of features (batch, seq_len, features)
            
        Returns:
            Temporal consistency loss value
        """
        if sequence.shape[1] < self.window_size:
            return torch.tensor(0.0, device=sequence.device)
        
        consistency_loss = 0.0
        count = 0
        
        # Check consistency within sliding windows
        for i in range(sequence.shape[1] - self.window_size + 1):
            window = sequence[:, i:i+self.window_size, :]
            
            # Compare consecutive frames within window
            for j in range(self.window_size - 1):
                frame_diff = self.mse_loss(window[:, j, :], window[:, j+1, :])
                consistency_loss += frame_diff
                count += 1
        
        if count > 0:
            consistency_loss /= count
        
        return self.weight * consistency_loss


class PathologySpecificLoss(nn.Module):
    """
    Loss function that applies different penalties based on pathology type.
    """
    
    def __init__(self, pathology_weights: Optional[Dict[int, float]] = None):
        """
        Initialize the pathology-specific loss.
        
        Args:
            pathology_weights: Dictionary mapping pathology indices to loss weights
        """
        super().__init__()
        
        # Default weights: higher penalty for more severe pathologies
        self.pathology_weights = pathology_weights or {
            0: 1.0,    # HC (Healthy Control) - baseline
            1: 1.2,    # Hip - moderate
            2: 1.3,    # Knee - moderate
            3: 1.4,    # Ankle - moderate to severe
            4: 1.5     # Calcaneus - severe
        }
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        logger.info(f"Initialized PathologySpecificLoss with weights: {self.pathology_weights}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pathology-specific loss.
        
        Args:
            predictions: Model predictions
            targets: True labels
            
        Returns:
            Weighted loss value
        """
        # Compute base cross-entropy loss
        base_loss = self.ce_loss(predictions, targets)
        
        # Apply pathology-specific weights
        weights = torch.tensor([self.pathology_weights.get(t.item(), 1.0) 
                              for t in targets], device=predictions.device)
        
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()


class CombinedGaitLoss(nn.Module):
    """
    Comprehensive loss function combining all gait-specific losses.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the combined gait loss.
        
        Args:
            config: Configuration dictionary for loss weights
        """
        super().__init__()
        
        # Default configuration
        default_config = {
            'classification_weight': 0.4,
            'progression_weight': 0.3,
            'asymmetry_weight': 0.1,
            'temporal_consistency_weight': 0.1,
            'pathology_specific_weight': 0.1
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Initialize individual loss functions
        self.progression_loss = GaitProgressionLoss()
        self.asymmetry_loss = AsymmetryLoss(weight=self.config['asymmetry_weight'])
        self.temporal_loss = TemporalConsistencyLoss(weight=self.config['temporal_consistency_weight'])
        self.pathology_loss = PathologySpecificLoss()
        
        logger.info(f"Initialized CombinedGaitLoss with config: {self.config}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined gait loss.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        total_loss = 0.0
        loss_components = {}
        
        # Classification loss
        if 'classification' in predictions and 'labels' in targets:
            class_loss = self.pathology_loss(predictions['classification'], targets['labels'])
            total_loss += self.config['classification_weight'] * class_loss
            loss_components['classification'] = class_loss.item()
        
        # Progression loss
        if 'progression' in predictions and 'next_state' in targets:
            prog_loss, prog_components = self.progression_loss(
                predictions['classification'], predictions['progression'],
                targets['labels'], targets['next_state']
            )
            total_loss += self.config['progression_weight'] * prog_loss
            loss_components['progression'] = prog_components['total']
        
        # Asymmetry loss
        if 'left_features' in predictions and 'right_features' in predictions:
            asym_loss = self.asymmetry_loss(predictions['left_features'], predictions['right_features'])
            total_loss += asym_loss
            loss_components['asymmetry'] = asym_loss.item()
        
        # Temporal consistency loss
        if 'sequence' in predictions:
            temp_loss = self.temporal_loss(predictions['sequence'])
            total_loss += temp_loss
            loss_components['temporal_consistency'] = temp_loss.item()
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components

