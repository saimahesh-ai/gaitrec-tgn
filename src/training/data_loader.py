import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from tqdm import tqdm

from ..models import GaitGraphBuilder
from ..preprocessing import GaitDataPreprocessor

logger = logging.getLogger(__name__)

def create_graph_dataloader(data_subset: pd.DataFrame, encoder, graph_builder: GaitGraphBuilder, 
                           batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader from GaitRec data.
    
    Args:
        data_subset: DataFrame containing metadata for the data subset
        encoder: Gait state encoder model
        graph_builder: Graph builder instance
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch Geometric DataLoader
    """
    graphs = []
    
    # Group by subject
    subjects = data_subset.groupby('SUBJECT_ID')
    
    logger.info(f"Creating data loader for {len(subjects)} subjects")
    
    for subject_id, subject_data in subjects:
        try:
            # Get all sessions for this subject
            sessions = subject_data.groupby('SESSION_ID')
            
            embeddings_list = []
            session_metadata = []
            
            for session_id, session in sessions:
                # Load GRF data for this session
                grf_data = load_session_grf(subject_id, session_id, data_subset)
                
                if grf_data is not None:
                    # Extract embeddings
                    with torch.no_grad():
                        embeddings = encoder(torch.tensor(grf_data).float())
                    embeddings_list.append(embeddings)
                    
                    # Store session metadata
                    session_metadata.append({
                        'session_id': session_id,
                        'timestamp': session.iloc[0].get('timestamp', 0),
                        'metadata': session.iloc[0]
                    })
            
            if len(embeddings_list) > 0:
                # Build temporal graph for this subject
                subject_graph = graph_builder.build_subject_graph(session_metadata, embeddings_list)
                
                if subject_graph is not None:
                    # Add labels
                    pathology_map = {'HC': 0, 'H': 1, 'K': 2, 'A': 3, 'C': 4}
                    subject_graph.y = torch.tensor(
                        pathology_map[subject_data.iloc[0]['CLASS']]
                    )
                    
                    # Add subject ID for tracking
                    subject_graph.subject_id = torch.tensor([hash(subject_id) % 1000000])
                    
                    graphs.append(subject_graph)
                    
        except Exception as e:
            logger.warning(f"Failed to process subject {subject_id}: {e}")
            continue
    
    if len(graphs) == 0:
        logger.error("No valid graphs created!")
        return None
    
    logger.info(f"Successfully created {len(graphs)} graphs")
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def load_session_grf(subject_id: str, session_id: str, metadata: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Load and format GRF data for a specific session.
    
    Args:
        subject_id: Subject identifier
        session_id: Session identifier
        metadata: Metadata DataFrame
        
    Returns:
        Array of shape (n_channels, sequence_length) or None if loading fails
    """
    try:
        # This is a placeholder implementation
        # In practice, you would load the actual GRF data files
        
        # For now, create synthetic data for testing
        # Replace this with actual data loading logic
        
        # Get metadata row for this session
        session_data = metadata[
            (metadata['SUBJECT_ID'] == subject_id) & 
            (metadata['SESSION_ID'] == session_id)
        ]
        
        if len(session_data) == 0:
            logger.warning(f"No metadata found for subject {subject_id}, session {session_id}")
            return None
        
        # Create synthetic GRF data (replace with actual loading)
        # 10 channels: F_V_L, F_V_R, F_AP_L, F_AP_R, F_ML_L, F_ML_R, COP_AP_L, COP_AP_R, COP_ML_L, COP_ML_R
        n_channels = 10
        sequence_length = 101  # Normalized gait cycle
        
        # Generate realistic-looking GRF data
        grf_data = np.random.randn(n_channels, sequence_length) * 0.1
        
        # Add some structure to make it look more realistic
        # Vertical force should have peaks
        grf_data[0, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5  # Left
        grf_data[1, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5  # Right
        
        # Anterior-posterior should have smaller variations
        grf_data[2:4, :] *= 0.3
        
        # Medio-lateral should be even smaller
        grf_data[4:6, :] *= 0.2
        
        # Center of pressure should be smooth
        grf_data[6:, :] = np.cumsum(np.random.randn(4, sequence_length) * 0.01, axis=1)
        
        return grf_data
        
    except Exception as e:
        logger.error(f"Failed to load GRF data for subject {subject_id}, session {session_id}: {e}")
        return None


class GaitDataLoader:
    """
    Custom data loader for GaitRec data with caching and preprocessing.
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the GaitRec dataset
            cache_dir: Directory for caching preprocessed data
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.preprocessor = GaitDataPreprocessor(data_path=data_path)
        self.graph_builder = GaitGraphBuilder()
        
        # Load metadata
        self.metadata = self.preprocessor.load_metadata()
        
        logger.info(f"Initialized GaitDataLoader with {len(self.metadata)} records")
    
    def get_data_splits(self, test_size: float = 0.2, val_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Use predefined splits if available
        if 'TRAIN_VALIDATE_TEST' in self.metadata.columns:
            train_data = self.metadata[self.metadata['TRAIN_VALIDATE_TEST'] == 'TRAIN']
            val_data = self.metadata[self.metadata['TRAIN_VALIDATE_TEST'] == 'VALIDATE']
            test_data = self.metadata[self.metadata['TRAIN_VALIDATE_TEST'] == 'TEST']
            
            logger.info(f"Using predefined splits: Train={len(train_data)}, "
                       f"Val={len(val_data)}, Test={len(test_data)}")
            
            return train_data, val_data, test_data
        
        # Otherwise, create random splits
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test_data = train_test_split(
            self.metadata, test_size=test_size, random_state=random_state, 
            stratify=self.metadata['CLASS'] if 'CLASS' in self.metadata.columns else None
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val, test_size=val_ratio, random_state=random_state,
            stratify=train_val['CLASS'] if 'CLASS' in train_val.columns else None
        )
        
        logger.info(f"Created random splits: Train={len(train_data)}, "
                   f"Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                          test_data: pd.DataFrame, encoder, batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Create data loaders for all splits.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            encoder: Gait state encoder
            batch_size: Batch size
            
        Returns:
            Dictionary containing train, validation, and test data loaders
        """
        dataloaders = {}
        
        # Create train loader
        train_loader = create_graph_dataloader(
            train_data, encoder, self.graph_builder, batch_size, shuffle=True
        )
        if train_loader:
            dataloaders['train'] = train_loader
        
        # Create validation loader
        val_loader = create_graph_dataloader(
            val_data, encoder, self.graph_builder, batch_size, shuffle=False
        )
        if val_loader:
            dataloaders['val'] = val_loader
        
        # Create test loader
        test_loader = create_graph_dataloader(
            test_data, encoder, self.graph_builder, batch_size, shuffle=False
        )
        if test_loader:
            dataloaders['test'] = test_loader
        
        return dataloaders
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}
        
        # Basic counts
        stats['total_records'] = len(self.metadata)
        stats['unique_subjects'] = self.metadata['SUBJECT_ID'].nunique()
        stats['unique_sessions'] = self.metadata['SESSION_ID'].nunique()
        
        # Pathology distribution
        if 'CLASS' in self.metadata.columns:
            class_counts = self.metadata['CLASS'].value_counts()
            stats['pathology_distribution'] = class_counts.to_dict()
        
        # Demographics
        if 'SEX' in self.metadata.columns:
            stats['sex_distribution'] = self.metadata['SEX'].value_counts().to_dict()
        
        if 'AGE' in self.metadata.columns:
            stats['age_stats'] = {
                'mean': self.metadata['AGE'].mean(),
                'std': self.metadata['AGE'].std(),
                'min': self.metadata['AGE'].min(),
                'max': self.metadata['AGE'].max()
            }
        
        if 'BODY_MASS' in self.metadata.columns:
            stats['body_mass_stats'] = {
                'mean': self.metadata['BODY_MASS'].mean(),
                'std': self.metadata['BODY_MASS'].std(),
                'min': self.metadata['BODY_MASS'].min(),
                'max': self.metadata['BODY_MASS'].max()
            }
        
        # Split distribution
        if 'TRAIN_VALIDATE_TEST' in self.metadata.columns:
            stats['split_distribution'] = self.metadata['TRAIN_VALIDATE_TEST'].value_counts().to_dict()
        
        return stats
    
    def preprocess_and_cache(self, force_reprocess: bool = False) -> str:
        """
        Preprocess all data and cache results.
        
        Args:
            force_reprocess: Whether to force reprocessing even if cache exists
            
        Returns:
            Path to cache directory
        """
        cache_file = self.cache_dir / "preprocessed_data.pkl"
        
        if not force_reprocess and cache_file.exists():
            logger.info("Loading preprocessed data from cache")
            return str(cache_file)
        
        logger.info("Preprocessing data and creating cache...")
        
        # Process all subjects
        processed_data = {}
        
        subjects = self.metadata.groupby('SUBJECT_ID')
        for subject_id, subject_data in tqdm(subjects, desc="Preprocessing subjects"):
            try:
                processed_subject = self.preprocessor.preprocess_subject(subject_id, subject_data)
                if processed_subject:
                    processed_data[subject_id] = processed_subject
            except Exception as e:
                logger.warning(f"Failed to preprocess subject {subject_id}: {e}")
                continue
        
        # Save to cache
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Preprocessing completed. Cached {len(processed_data)} subjects to {cache_file}")
        
        return str(cache_file)
