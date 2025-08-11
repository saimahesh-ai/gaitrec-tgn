#!/usr/bin/env python3
"""
Main training script for the GaitRec Temporal Graph Network system.
This script demonstrates the complete pipeline from data loading to model training.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from models import GaitStateEncoder, TemporalGaitGNN
from training import GaitTGNTrainer, GaitDataLoader
from training.training_utils import TrainingConfig
from analysis import ClinicalGaitAnalyzer, GaitVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gaitrec_tgn_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(config_path: str = None) -> TrainingConfig:
    """
    Setup training configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration object
    """
    if config_path and os.path.exists(config_path):
        config = TrainingConfig.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = TrainingConfig()
        logger.info("Using default configuration")
    
    # Save configuration
    config.save('config.yaml')
    
    return config

def create_models(config: TrainingConfig):
    """
    Create and initialize models.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (state_encoder, tgn_model)
    """
    logger.info("Creating models...")
    
    # Create state encoder
    state_encoder = GaitStateEncoder(
        input_channels=config.input_channels,
        embedding_dim=config.embedding_dim,
        sequence_length=101  # Normalized gait cycle
    )
    
    # Create TGN model
    tgn_model = TemporalGaitGNN(
        input_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_classes=config.num_classes
    )
    
    logger.info(f"State encoder parameters: {sum(p.numel() for p in state_encoder.parameters()):,}")
    logger.info(f"TGN model parameters: {sum(p.numel() for p in tgn_model.parameters()):,}")
    
    return state_encoder, tgn_model

def prepare_data(config: TrainingConfig):
    """
    Prepare data loaders.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary of data loaders
    """
    logger.info("Preparing data...")
    
    # Initialize data loader
    data_loader = GaitDataLoader(
        data_path=config.data_path,
        cache_dir=config.cache_dir
    )
    
    # Get dataset statistics
    stats = data_loader.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    # Get data splits
    train_data, val_data, test_data = data_loader.get_data_splits()
    
    # Create data loaders
    dataloaders = data_loader.create_dataloaders(
        train_data, val_data, test_data,
        encoder=None,  # Will be set later
        batch_size=config.batch_size
    )
    
    if not dataloaders:
        raise RuntimeError("Failed to create data loaders")
    
    logger.info(f"Created data loaders: {list(dataloaders.keys())}")
    
    return dataloaders, data_loader

def train_models(state_encoder, tgn_model, dataloaders, config: TrainingConfig):
    """
    Train the models.
    
    Args:
        state_encoder: Gait state encoder
        tgn_model: Temporal Graph Network
        dataloaders: Dictionary of data loaders
        config: Training configuration
        
    Returns:
        Training history
    """
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = GaitTGNTrainer(
        state_encoder=state_encoder,
        tgn_model=tgn_model,
        device='auto',
        config=config.to_dict()
    )
    
    # Train models
    training_history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val']
    )
    
    logger.info("Training completed!")
    
    return training_history, trainer

def evaluate_models(trainer, dataloaders):
    """
    Evaluate the trained models.
    
    Args:
        trainer: Trained trainer instance
        dataloaders: Dictionary of data loaders
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating models...")
    
    # Evaluate on test set
    test_results = trainer.evaluate(dataloaders['test'])
    
    logger.info(f"Test Results: {test_results}")
    
    return test_results

def run_clinical_analysis(model_path: str, sample_data: dict):
    """
    Run clinical analysis on sample data.
    
    Args:
        model_path: Path to trained model
        sample_data: Sample patient data
        
    Returns:
        Clinical analysis results
    """
    logger.info("Running clinical analysis...")
    
    try:
        # Initialize clinical analyzer
        analyzer = ClinicalGaitAnalyzer(model_path)
        
        # Assess patient
        report = analyzer.assess_patient(sample_data)
        
        logger.info("Clinical analysis completed!")
        logger.info(f"Patient ID: {report['patient_id']}")
        logger.info(f"Overall Risk: {report['overall_risk']['level']}")
        logger.info(f"Recommendations: {len(report['recommendations'])}")
        
        return report
        
    except Exception as e:
        logger.error(f"Clinical analysis failed: {e}")
        return None

def create_sample_data():
    """
    Create sample patient data for demonstration.
    
    Returns:
        Sample patient data dictionary
    """
    # Create synthetic GRF data for demonstration
    n_channels = 10
    sequence_length = 101
    
    sample_data = {
        'id': 'PATIENT_001',
        'sessions': [
            {
                'id': 'session_1',
                'timestamp': 0,
                'grf_data': np.random.randn(n_channels, sequence_length) * 0.1,
                'clinical_score': 75.0
            },
            {
                'id': 'session_2', 
                'timestamp': 7,  # 1 week later
                'grf_data': np.random.randn(n_channels, sequence_length) * 0.1,
                'clinical_score': 78.0
            }
        ]
    }
    
    # Add some structure to make it look realistic
    for session in sample_data['sessions']:
        grf_data = session['grf_data']
        # Vertical force should have peaks
        grf_data[0, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5
        grf_data[1, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5
        
        # Other forces
        grf_data[2:4, :] *= 0.3
        grf_data[4:6, :] *= 0.2
        grf_data[6:, :] = np.cumsum(np.random.randn(4, sequence_length) * 0.01, axis=1)
    
    return sample_data

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='GaitRec TGN Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run analysis')
    
    args = parser.parse_args()
    
    try:
        # Setup configuration
        config = setup_experiment(args.config)
        
        # Override with command line arguments
        if args.data_path:
            config.data_path = args.data_path
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        
        logger.info(f"Configuration: {config.to_dict()}")
        
        if not args.skip_training:
            # Create models
            state_encoder, tgn_model = create_models(config)
            
            # Prepare data
            dataloaders, data_loader = prepare_data(config)
            
            # Update data loaders with encoder
            dataloaders = data_loader.create_dataloaders(
                data_loader.get_data_splits()[0],  # train
                data_loader.get_data_splits()[1],  # val
                data_loader.get_data_splits()[2],  # test
                encoder=state_encoder,
                batch_size=config.batch_size
            )
            
            # Train models
            training_history, trainer = train_models(
                state_encoder, tgn_model, dataloaders, config
            )
            
            # Evaluate models
            test_results = evaluate_models(trainer, dataloaders)
            
            logger.info("Training pipeline completed successfully!")
            
            # Save training history
            import json
            with open('training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)
            
        else:
            logger.info("Skipping training as requested")
        
        # Run clinical analysis if model exists
        model_path = Path(config.save_dir) / 'best_model.pth'
        if model_path.exists():
            logger.info("Running clinical analysis...")
            
            # Create sample data
            sample_data = create_sample_data()
            
            # Run analysis
            clinical_report = run_clinical_analysis(str(model_path), sample_data)
            
            if clinical_report:
                # Save clinical report
                import json
                with open('clinical_report.json', 'w') as f:
                    # Convert datetime to string for JSON serialization
                    report_copy = clinical_report.copy()
                    report_copy['timestamp'] = str(report_copy['timestamp'])
                    json.dump(report_copy, f, indent=2)
                
                logger.info("Clinical analysis completed and saved!")
        else:
            logger.warning("No trained model found for clinical analysis")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
