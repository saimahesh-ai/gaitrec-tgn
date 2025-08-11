#!/usr/bin/env python3
"""
Demo script for the GaitRec Temporal Graph Network system.
This script demonstrates the system with synthetic data for testing purposes.
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.models import GaitStateEncoder, TemporalGaitGNN, GaitGraphBuilder
from src.training import GaitTGNTrainer
from src.analysis import ClinicalGaitAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_dataset(num_subjects=10, num_sessions=3):
    """
    Create synthetic GaitRec dataset for demonstration.
    
    Args:
        num_subjects: Number of subjects to create
        num_sessions: Number of sessions per subject
        
    Returns:
        Dictionary containing synthetic dataset
    """
    logger.info(f"Creating synthetic dataset with {num_subjects} subjects, {num_sessions} sessions each")
    
    dataset = {
        'metadata': [],
        'grf_data': {}
    }
    
    # Pathology classes
    pathologies = ['HC', 'H', 'K', 'A', 'C']
    
    for subject_id in range(num_subjects):
        # Randomly assign pathology
        pathology = np.random.choice(pathologies, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        for session_id in range(num_sessions):
            # Create metadata entry
            metadata_entry = {
                'SUBJECT_ID': f'SUBJECT_{subject_id:03d}',
                'SESSION_ID': f'SESSION_{session_id:03d}',
                'CLASS': pathology,
                'SEX': np.random.choice(['M', 'F']),
                'AGE': np.random.randint(20, 80),
                'BODY_MASS': np.random.uniform(50, 100),
                'TRAIN_VALIDATE_TEST': np.random.choice(['TRAIN', 'VALIDATE', 'TEST'], p=[0.7, 0.15, 0.15])
            }
            
            dataset['metadata'].append(metadata_entry)
            
            # Create synthetic GRF data
            n_channels = 10
            sequence_length = 101
            
            # Generate realistic GRF data
            grf_data = np.random.randn(n_channels, sequence_length) * 0.1
            
            # Add structure based on pathology
            if pathology == 'HC':
                # Healthy controls have more symmetric patterns
                grf_data[0, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5
                grf_data[1, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * 0.5
            else:
                # Pathological patterns have more asymmetry
                asymmetry_factor = 0.3
                grf_data[0, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * (0.5 + asymmetry_factor)
                grf_data[1, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length))) * (0.5 - asymmetry_factor)
            
            # Other force components
            grf_data[2:4, :] *= 0.3
            grf_data[4:6, :] *= 0.2
            grf_data[6:, :] = np.cumsum(np.random.randn(4, sequence_length) * 0.01, axis=1)
            
            # Store GRF data
            key = f"{metadata_entry['SUBJECT_ID']}_{metadata_entry['SESSION_ID']}"
            dataset['grf_data'][key] = grf_data
    
    logger.info(f"Created {len(dataset['metadata'])} metadata entries")
    logger.info(f"Created {len(dataset['grf_data'])} GRF data samples")
    
    return dataset

def test_model_creation():
    """Test model creation and basic forward pass."""
    logger.info("Testing model creation...")
    
    try:
        # Create models
        encoder = GaitStateEncoder(input_channels=10, embedding_dim=128)
        tgn_model = TemporalGaitGNN(input_dim=128, hidden_dim=256, output_dim=64, num_classes=5)
        
        # Create synthetic input
        batch_size = 4
        input_data = torch.randn(batch_size, 10, 101)  # (batch, channels, sequence)
        
        # Test encoder
        with torch.no_grad():
            embeddings = encoder(input_data)
            logger.info(f"Encoder output shape: {embeddings.shape}")
        
        # Test TGN model
        graph_builder = GaitGraphBuilder()
        graph = graph_builder.build_subject_graph(
            [{'session_id': 'test', 'timestamp': 0}],
            [embeddings]
        )
        
        if graph is not None:
            with torch.no_grad():
                classification, progression = tgn_model(
                    graph.x, graph.edge_index, graph.timestamps
                )
                logger.info(f"TGN classification output shape: {classification.shape}")
                logger.info(f"TGN progression output shape: {progression.shape}")
        
        logger.info("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {e}")
        return False

def test_training_pipeline():
    """Test the training pipeline with synthetic data."""
    logger.info("Testing training pipeline...")
    
    try:
        # Create synthetic dataset
        dataset = create_synthetic_dataset(num_subjects=20, num_sessions=2)
        
        # Create models
        encoder = GaitStateEncoder(input_channels=10, embedding_dim=64)  # Smaller for testing
        tgn_model = TemporalGaitGNN(input_dim=64, hidden_dim=128, output_dim=32, num_classes=5)
        
        # Create trainer
        config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 8,
            'num_epochs': 5,  # Very short for testing
            'save_dir': 'demo_checkpoints'
        }
        
        trainer = GaitTGNTrainer(
            state_encoder=encoder,
            tgn_model=tgn_model,
            device='cpu',  # Use CPU for demo
            config=config
        )
        
        logger.info("✅ Training pipeline test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline test failed: {e}")
        return False

def test_clinical_analysis():
    """Test clinical analysis functionality."""
    logger.info("Testing clinical analysis...")
    
    try:
        # Create sample patient data
        sample_data = {
            'id': 'DEMO_PATIENT_001',
            'sessions': [
                {
                    'id': 'session_1',
                    'timestamp': 0,
                    'grf_data': np.random.randn(10, 101) * 0.1,
                    'clinical_score': 70.0
                }
            ]
        }
        
        # Add some structure
        grf_data = sample_data['sessions'][0]['grf_data']
        grf_data[0, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, 101))) * 0.5
        grf_data[1, :] += np.abs(np.sin(np.linspace(0, 2*np.pi, 101))) * 0.5
        
        logger.info("✅ Clinical analysis test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clinical analysis test failed: {e}")
        return False

def run_demo():
    """Run the complete demo."""
    logger.info("🚀 Starting GaitRec TGN Demo")
    logger.info("=" * 50)
    
    # Test 1: Model Creation
    logger.info("\n📋 Test 1: Model Creation")
    test1_passed = test_model_creation()
    
    # Test 2: Training Pipeline
    logger.info("\n📋 Test 2: Training Pipeline")
    test2_passed = test_training_pipeline()
    
    # Test 3: Clinical Analysis
    logger.info("\n📋 Test 3: Clinical Analysis")
    test3_passed = test_clinical_analysis()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Demo Results Summary")
    logger.info("=" * 50)
    
    tests = [
        ("Model Creation", test1_passed),
        ("Training Pipeline", test2_passed),
        ("Clinical Analysis", test3_passed)
    ]
    
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_passed = sum([passed for _, passed in tests])
    total_tests = len(tests)
    
    logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("🎉 All tests passed! The system is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the error messages above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run demo
    success = run_demo()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
