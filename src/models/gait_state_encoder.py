import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GaitStateEncoder(nn.Module):
    """
    CNN-LSTM architecture for encoding gait cycles into state embeddings.
    Combines convolutional layers for local pattern extraction with LSTM for temporal dependencies.
    """
    
    def __init__(self, input_channels: int = 10, embedding_dim: int = 128, 
                 sequence_length: int = 101, dropout: float = 0.2):
        """
        Initialize the gait state encoder.
        
        Args:
            input_channels: Number of input channels (force components)
            embedding_dim: Dimension of the output embedding
            sequence_length: Length of the normalized gait cycle sequence
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        # CNN for local pattern extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(128, 256, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Final embedding layer
        self.fc = nn.Linear(512, embedding_dim)  # 512 = 256 * 2 (bidirectional)
        self.final_bn = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized GaitStateEncoder with {input_channels} input channels, "
                   f"{embedding_dim} embedding dimension")
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutions and pooling."""
        size = self.sequence_length
        
        # After conv1 + pool
        size = size // 2
        
        # After conv2 + pool
        size = size // 2
        
        # After conv3 (no pooling here)
        return size
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.size(0)
        
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {x.size(1)}")
        if x.size(2) != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {x.size(2)}")
        
        try:
            # CNN layers with batch normalization and ReLU
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout(x)
            
            # Reshape for LSTM: (batch, sequence, features)
            x = x.transpose(1, 2)
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Global average pooling over the sequence dimension
            x = torch.mean(lstm_out, dim=1)
            
            # Final embedding layer
            embedding = self.fc(x)
            embedding = self.final_bn(embedding)
            
            # Apply L2 normalization for stable training
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return zero embedding in case of error
            return torch.zeros(batch_size, self.embedding_dim, device=x.device)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def get_input_channels(self) -> int:
        """Get the number of input channels."""
        return self.input_channels
    
    def get_sequence_length(self) -> int:
        """Get the expected sequence length."""
        return self.sequence_length


class GaitStateEncoderV2(nn.Module):
    """
    Enhanced version of the gait state encoder with attention mechanism.
    """
    
    def __init__(self, input_channels: int = 10, embedding_dim: int = 128, 
                 sequence_length: int = 101, dropout: float = 0.2, 
                 num_heads: int = 8):
        """
        Initialize the enhanced gait state encoder.
        
        Args:
            input_channels: Number of input channels (force components)
            embedding_dim: Dimension of the output embedding
            sequence_length: Length of the normalized gait cycle sequence
            dropout: Dropout rate for regularization
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.num_heads = num_heads
        
        # CNN layers (same as basic version)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(128, 256, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Final embedding layer
        self.fc = nn.Linear(512, embedding_dim)
        self.final_bn = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized GaitStateEncoderV2 with attention mechanism")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.size(0)
        
        try:
            # CNN layers
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout(x)
            
            # Reshape for attention: (batch, sequence, features)
            x = x.transpose(1, 2)
            
            # Self-attention mechanism
            x_attended, attention_weights = self.attention(x, x, x)
            x = x + x_attended  # Residual connection
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Global average pooling
            x = torch.mean(lstm_out, dim=1)
            
            # Final embedding
            embedding = self.fc(x)
            embedding = self.final_bn(embedding)
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            return torch.zeros(batch_size, self.embedding_dim, device=x.device)


class GaitStateEncoderV3(nn.Module):
    """
    Lightweight version of the gait state encoder for real-time applications.
    """
    
    def __init__(self, input_channels: int = 10, embedding_dim: int = 64, 
                 sequence_length: int = 101, dropout: float = 0.1):
        """
        Initialize the lightweight gait state encoder.
        
        Args:
            input_channels: Number of input channels
            embedding_dim: Dimension of the output embedding (smaller for efficiency)
            sequence_length: Length of the normalized gait cycle sequence
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        # Simplified CNN architecture
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(dropout)
        
        # Simple feedforward network
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized lightweight GaitStateEncoderV3")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the lightweight encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.size(0)
        
        try:
            # Simplified CNN layers
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            
            # Global average pooling
            x = self.pool(x).squeeze(-1)
            
            # Feedforward layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            embedding = self.fc2(x)
            
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            return torch.zeros(batch_size, self.embedding_dim, device=x.device)
