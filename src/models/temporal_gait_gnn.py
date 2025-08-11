import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GaitGraphBuilder:
    """
    Builds temporal graphs for gait analysis from gait cycle embeddings.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the graph builder.
        
        Args:
            similarity_threshold: Threshold for similarity-based edge creation
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized GaitGraphBuilder with similarity threshold {similarity_threshold}")
    
    def build_subject_graph(self, subject_sessions: List[Dict], 
                           embeddings: List[torch.Tensor]) -> Data:
        """
        Build temporal graph for a single subject across sessions.
        
        Args:
            subject_sessions: List of session metadata dictionaries
            embeddings: List of embeddings for each session
            
        Returns:
            PyTorch Geometric Data object representing the temporal graph
        """
        nodes = []
        edges = []
        edge_attr = []
        node_timestamps = []
        
        # Create nodes for each gait cycle
        for session_idx, session in enumerate(subject_sessions):
            for cycle_idx, embedding in enumerate(embeddings[session_idx]):
                nodes.append(embedding.detach().cpu().numpy())
                # Use session index as timestamp for now (can be enhanced with actual timestamps)
                node_timestamps.append(session_idx)
        
        if len(nodes) == 0:
            logger.warning("No embeddings provided for graph construction")
            return None
            
        nodes = np.array(nodes)
        
        # Create edges between temporally adjacent cycles
        for i in range(len(nodes) - 1):
            if node_timestamps[i] == node_timestamps[i + 1]:
                # Same session - strong connection
                edges.append([i, i + 1])
                edge_attr.append(1.0)
            else:
                # Different sessions - weaker connection
                time_diff = node_timestamps[i + 1] - node_timestamps[i]
                weight = np.exp(-time_diff / 10)  # Decay over sessions
                edges.append([i, i + 1])
                edge_attr.append(weight)
        
        # Add similarity-based edges
        similarity_edges = self._add_similarity_edges(nodes)
        edges.extend(similarity_edges)
        
        if len(edges) == 0:
            logger.warning("No edges created in graph")
            return None
            
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = torch.tensor(nodes, dtype=torch.float)
        timestamps = torch.tensor(node_timestamps, dtype=torch.long)
        
        logger.info(f"Built subject graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            timestamps=timestamps
        )
    
    def _add_similarity_edges(self, embeddings: np.ndarray) -> List[List[int]]:
        """Add edges based on embedding similarity."""
        if len(embeddings) < 2:
            return []
            
        similarity_matrix = cosine_similarity(embeddings)
        edges = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    
        logger.info(f"Added {len(edges)} similarity-based edges")
        return edges
    
    def build_population_graph(self, all_subjects_data: Dict) -> Data:
        """
        Build graph connecting similar gait patterns across population.
        
        Args:
            all_subjects_data: Dictionary mapping subject_id to subject data
            
        Returns:
            PyTorch Geometric Data object representing the population graph
        """
        # Aggregate all embeddings
        all_embeddings = []
        subject_labels = []
        pathology_labels = []
        
        for subject_id, data in all_subjects_data.items():
            if 'embeddings' in data and len(data['embeddings']) > 0:
                all_embeddings.extend(data['embeddings'])
                subject_labels.extend([subject_id] * len(data['embeddings']))
                pathology_labels.extend([data.get('pathology', 0)] * len(data['embeddings']))
        
        if len(all_embeddings) == 0:
            logger.warning("No embeddings found for population graph")
            return None
            
        all_embeddings = np.array(all_embeddings)
        
        # Build k-NN graph
        try:
            knn_graph = kneighbors_graph(
                all_embeddings, n_neighbors=min(10, len(all_embeddings)-1), 
                mode='distance'
            )
            edges = np.array(knn_graph.nonzero()).T
        except Exception as e:
            logger.warning(f"Failed to build k-NN graph: {e}")
            edges = np.array([])
        
        logger.info(f"Built population graph with {len(all_embeddings)} nodes and {len(edges)} edges")
        
        return Data(
            x=torch.tensor(all_embeddings, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long) if len(edges) > 0 else torch.empty((2, 0), dtype=torch.long),
            subject_id=torch.tensor(subject_labels, dtype=torch.long),
            pathology=torch.tensor(pathology_labels, dtype=torch.long)
        )


class TemporalGaitGNN(nn.Module):
    """
    Temporal Graph Neural Network for gait progression modeling.
    Combines graph convolutions with temporal attention mechanisms.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, 
                 output_dim: int = 64, num_classes: int = 5, 
                 dropout: float = 0.3):
        """
        Initialize the Temporal Gait GNN.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_classes: Number of pathology classes (HC, Hip, Knee, Ankle, Calcaneus)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, dropout=dropout
        )
        
        # GRU for temporal dynamics
        self.gru = nn.GRU(output_dim, hidden_dim, num_layers=2, 
                         batch_first=True, bidirectional=True, dropout=dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Progression prediction head
        self.progression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized TemporalGaitGNN with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}, "
                   f"num_classes={num_classes}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Node features tensor
            edge_index: Graph connectivity
            timestamps: Optional temporal information
            batch: Optional batch assignment for graph-level tasks
            
        Returns:
            Tuple of (classification_output, progression_output)
        """
        # Graph convolutions with residual connections
        identity = x
        
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        
        # Apply temporal attention if timestamps provided
        if timestamps is not None and len(timestamps) > 1:
            try:
                # Sort by timestamp for temporal processing
                sorted_idx = torch.argsort(timestamps)
                x_temporal = x[sorted_idx]
                
                # Self-attention over temporal sequence
                x_attended, _ = self.temporal_attention(
                    x_temporal.unsqueeze(0), 
                    x_temporal.unsqueeze(0), 
                    x_temporal.unsqueeze(0)
                )
                x = x_attended.squeeze(0)
                
                # GRU for temporal modeling
                x_gru, _ = self.gru(x.unsqueeze(0))
                x = x_gru.squeeze(0)
                
                # Restore original order
                inverse_idx = torch.argsort(sorted_idx)
                x = x[inverse_idx]
                
            except Exception as e:
                logger.warning(f"Temporal processing failed: {e}, using graph-only features")
        
        # Global pooling for graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # If no batch, use mean pooling over nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Dual outputs
        classification = self.classifier(x)
        progression = self.progression_head(x)
        
        return classification, progression
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim
    
    def get_num_classes(self) -> int:
        """Get the number of output classes."""
        return self.num_classes


class GaitProgressionPredictor(nn.Module):
    """
    Specialized model for predicting gait progression over time.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, 
                 num_timesteps: int = 10):
        """
        Initialize the progression predictor.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_timesteps: Number of future timesteps to predict
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # LSTM for temporal progression modeling
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.2)
        
        # Progression prediction head
        self.progression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, input_dim * num_timesteps)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_timesteps)
        )
        
        logger.info(f"Initialized GaitProgressionPredictor with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, num_timesteps={num_timesteps}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict future gait states.
        
        Args:
            x: Input sequence tensor of shape (batch, seq_len, features)
            
        Returns:
            Tuple of (progression_predictions, confidence_scores)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        
        # Predict progression
        progression = self.progression_head(last_hidden)
        progression = progression.view(-1, self.num_timesteps, self.input_dim)
        
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_head(last_hidden))
        
        return progression, confidence

