<<<<<<< HEAD
# GaitRec Temporal Graph Network (TGN) System

A comprehensive implementation of Temporal Graph Networks for analyzing the GaitRec dataset, enabling pathology detection, progression tracking, and clinical decision support for musculoskeletal gait impairments.

## 🚀 Features

- **Advanced Gait Analysis**: CNN-LSTM based gait state encoding with temporal attention mechanisms
- **Temporal Graph Networks**: Graph-based modeling of gait progression over time
- **Multi-Pathology Classification**: Detection of Hip, Knee, Ankle, and Calcaneus impairments
- **Progression Prediction**: Forecasting future gait states for rehabilitation planning
- **Clinical Decision Support**: Risk assessment and treatment recommendations
- **Comprehensive Training Pipeline**: End-to-end training with advanced loss functions
- **Real-time Monitoring**: Clinical analysis tools for patient assessment

## 📁 Project Structure

```
gaitrec-tgn/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   ├── gait_state_encoder.py # CNN-LSTM gait encoder
│   │   ├── temporal_gait_gnn.py  # Temporal Graph Network
│   │   └── losses.py            # Custom loss functions
│   ├── preprocessing/            # Data preprocessing
│   │   ├── gait_preprocessor.py # GRF data preprocessing
│   │   └── feature_extractor.py # Feature extraction
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py           # Main trainer class
│   │   ├── data_loader.py       # Graph data loading
│   │   └── training_utils.py    # Training utilities
│   └── analysis/                # Clinical analysis tools
│       └── gait_analyzer.py     # Clinical decision support
├── data/                        # Dataset directory
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Python dependencies
├── main.py                     # Main training script
├── demo.py                     # Demo script for testing
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd gaitrec-tgn
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python demo.py
```

## 📊 Dataset

The system is designed for the **GaitRec dataset**, which contains:
- 75,732 bilateral walking trials
- 2,084 patients with musculoskeletal impairments
- 211 healthy controls
- 5 pathology classes: HC (Healthy), H (Hip), K (Knee), A (Ankle), C (Calcaneus)

### Data Structure
```
data/
├── GRF-metadata.csv           # Patient metadata and labels
├── GRF_F_V-RAW_LEFT.csv      # Left vertical force data
├── GRF_F_V-RAW_RIGHT.csv     # Right vertical force data
├── GRF_F_AP-RAW_LEFT.csv     # Left anterior-posterior force
├── GRF_F_AP-RAW_RIGHT.csv    # Right anterior-posterior force
├── GRF_F_ML-RAW_LEFT.csv     # Left medio-lateral force
├── GRF_F_ML-RAW_RIGHT.csv    # Right medio-lateral force
├── GRF_COP_AP-RAW_LEFT.csv   # Left center of pressure AP
├── GRF_COP_AP-RAW_RIGHT.csv  # Right center of pressure AP
├── GRF_COP_ML-RAW_LEFT.csv   # Left center of pressure ML
└── GRF_COP_ML-RAW_RIGHT.csv  # Right center of pressure ML
```

## 🏗️ Architecture

### 1. Gait State Encoder
- **CNN Layers**: Extract local force patterns from GRF signals
- **LSTM Layers**: Capture temporal dependencies across gait cycles
- **Attention Mechanism**: Focus on relevant gait phases
- **Output**: 128-dimensional gait state embeddings

### 2. Temporal Graph Network
- **Graph Construction**: Temporal connections between gait cycles
- **Graph Convolutions**: Learn spatial relationships in gait patterns
- **Temporal Attention**: Model progression over time
- **Dual Outputs**: Pathology classification + progression prediction

### 3. Advanced Loss Functions
- **Classification Loss**: Pathology detection accuracy
- **Progression Loss**: Temporal consistency in predictions
- **Asymmetry Loss**: Encourage symmetric gait patterns
- **Temporal Consistency**: Smooth progression over time

## 🚀 Usage

### Quick Start

1. **Run the demo** to test the system:
```bash
python demo.py
```

2. **Train the model** with default settings:
```bash
python src/main.py --epochs 100 --batch_size 32
```

3. **Custom training** with configuration:
```bash
python src/main.py --config config.yaml --data_path /path/to/data
```

### Training Configuration

Create a `config.yaml` file:
```yaml
# Model parameters
input_channels: 10
embedding_dim: 128
hidden_dim: 256
output_dim: 64
num_classes: 5

# Training parameters
learning_rate: 0.001
weight_decay: 1e-5
batch_size: 32
num_epochs: 100

# Loss weights
classification_weight: 0.4
progression_weight: 0.3
asymmetry_weight: 0.1
temporal_consistency_weight: 0.1
pathology_specific_weight: 0.1
```

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --config PATH        Configuration file path
  --data_path PATH    Dataset directory path
  --epochs INT        Number of training epochs
  --batch_size INT    Training batch size
  --lr FLOAT          Learning rate
  --skip_training     Skip training, run analysis only
```

## 🔬 Clinical Applications

### 1. Pathology Detection
- **Multi-class Classification**: Identify specific musculoskeletal impairments
- **Confidence Scoring**: Assess prediction reliability
- **Risk Stratification**: Categorize patients by severity

### 2. Progression Tracking
- **Recovery Monitoring**: Track rehabilitation progress over time
- **Deterioration Detection**: Identify worsening conditions early
- **Treatment Response**: Evaluate intervention effectiveness

### 3. Clinical Decision Support
- **Risk Assessment**: Composite risk scores for patient management
- **Treatment Recommendations**: Evidence-based intervention suggestions
- **Asymmetry Analysis**: Quantify left-right limb differences

### 4. Population Analysis
- **Pattern Recognition**: Identify common gait characteristics
- **Risk Signatures**: Discover high-risk patient subgroups
- **Comparative Analysis**: Benchmark against healthy controls

## 📈 Training and Evaluation

### Training Process
1. **Data Preprocessing**: GRF signal normalization and gait cycle segmentation
2. **Feature Extraction**: Advanced statistical and frequency-domain features
3. **Graph Construction**: Temporal and similarity-based graph building
4. **Model Training**: End-to-end training with custom loss functions
5. **Validation**: Regular evaluation on validation set
6. **Model Selection**: Save best performing models

### Evaluation Metrics
- **Classification**: Accuracy, F1-score, Precision, Recall
- **Progression**: Mean squared error, temporal consistency
- **Clinical**: Risk scores, asymmetry indices, recovery rates

### Model Checkpointing
- **Best Model**: Automatically save best validation performance
- **Training History**: Track metrics across epochs
- **Configuration**: Save training parameters for reproducibility

## 🧪 Testing and Validation

### Demo Script
The `demo.py` script provides comprehensive testing:
- **Model Creation**: Test neural network architectures
- **Training Pipeline**: Validate training infrastructure
- **Clinical Analysis**: Test clinical decision support

### Synthetic Data
- **Realistic GRF Patterns**: Generate pathological and healthy gait data
- **Multiple Subjects**: Test with various patient profiles
- **Temporal Sequences**: Validate progression modeling

## 🔧 Advanced Features

### 1. Mixed Precision Training
- **FP16 Training**: Reduce memory usage and speed up training
- **Gradient Scaling**: Maintain numerical stability

### 2. Advanced Optimizers
- **AdamW**: Improved weight decay implementation
- **Cosine Annealing**: Smooth learning rate scheduling
- **Gradient Clipping**: Prevent gradient explosion

### 3. Data Augmentation
- **Signal Perturbation**: Add realistic noise to GRF data
- **Temporal Shifts**: Simulate timing variations
- **Amplitude Scaling**: Model strength variations

### 4. Model Interpretability
- **Attention Visualization**: Understand model focus areas
- **Feature Importance**: Identify critical gait characteristics
- **Decision Paths**: Trace classification reasoning

## 📊 Results and Performance

### Expected Performance
- **Classification Accuracy**: 85-90% on pathology detection
- **Progression Prediction**: Low MSE for temporal forecasting
- **Clinical Risk Assessment**: High correlation with expert ratings

### Model Complexity
- **State Encoder**: ~2M parameters
- **TGN Model**: ~1.5M parameters
- **Total**: ~3.5M trainable parameters

### Training Time
- **CPU**: ~2-4 hours per epoch (100 epochs = 8-16 days)
- **GPU**: ~10-20 minutes per epoch (100 epochs = 1-3 days)

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **CUDA Issues**: Check PyTorch and CUDA compatibility
3. **Memory Errors**: Reduce batch size or use gradient accumulation
4. **Data Loading**: Verify dataset structure and file paths

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tuning
- **Batch Size**: Adjust based on available memory
- **Learning Rate**: Use learning rate finder for optimal values
- **Model Size**: Reduce dimensions for faster training

## 🔮 Future Enhancements

### Planned Features
- **Real-time Processing**: Live GRF data analysis
- **Wearable Integration**: Mobile and sensor data support
- **Multi-modal Fusion**: Combine GRF with video and EMG data
- **Personalized Models**: Patient-specific adaptation

### Research Directions
- **Self-supervised Learning**: Unsupervised gait representation learning
- **Transfer Learning**: Cross-dataset generalization
- **Causal Inference**: Understand treatment effects
- **Federated Learning**: Privacy-preserving multi-center training

## 📚 References

1. **GaitRec Dataset**: [Paper reference]
2. **Temporal Graph Networks**: [Paper reference]
3. **Gait Analysis**: [Review paper reference]
4. **Clinical Applications**: [Clinical study reference]

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GaitRec Dataset Contributors**: For providing the comprehensive gait dataset
- **PyTorch Geometric**: For excellent graph neural network support
- **Research Community**: For advancing gait analysis and temporal modeling

## 📞 Support

For questions and support:
- **Issues**: Use GitHub issues for bug reports
- **Discussions**: Join our community discussions
- **Email**: Contact the development team

---

**Note**: This system is designed for research and clinical applications. Always validate results with clinical experts and follow appropriate medical protocols.
=======
# gaitrec-tgn
GaitRec Temporal Gait Graph Neural Network project
>>>>>>> b8051cf1f0e36c66fe06de7769925013f554120c
