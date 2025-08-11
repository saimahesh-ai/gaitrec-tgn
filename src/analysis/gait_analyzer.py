import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path

from ..models import GaitStateEncoder, TemporalGaitGNN
from ..preprocessing import GaitDataPreprocessor

logger = logging.getLogger(__name__)

class GaitProgressionAnalyzer:
    """
    Analyzes gait progression over time for rehabilitation patients.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the progression analyzer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load trained models
        self._load_models(model_path)
        
        # Initialize preprocessor
        self.preprocessor = GaitDataPreprocessor()
        
        logger.info(f"Initialized GaitProgressionAnalyzer on device: {self.device}")
    
    def _load_models(self, model_path: str):
        """Load trained models from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize models
        self.encoder = GaitStateEncoder()
        self.tgn_model = TemporalGaitGNN()
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.tgn_model.load_state_dict(checkpoint['tgn_model_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.tgn_model.eval()
        
        # Move to device
        self.encoder.to(self.device)
        self.tgn_model.to(self.device)
        
        logger.info(f"Models loaded from {model_path}")
    
    def analyze_recovery_trajectory(self, patient_data: Dict) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Analyze recovery progression for rehabilitation patients.
        
        Args:
            patient_data: Dictionary containing patient gait data across sessions
            
        Returns:
            Tuple of (trajectories, metrics)
        """
        trajectories = []
        
        for session in patient_data['sessions']:
            try:
                # Get embeddings for each gait cycle
                grf_data = torch.tensor(session['grf_data'], dtype=torch.float).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.encoder(grf_data)
                    
                    # Build graph for this session
                    from ..models import GaitGraphBuilder
                    graph_builder = GaitGraphBuilder()
                    
                    # Create simple graph for single session
                    session_graph = graph_builder.build_subject_graph(
                        [{'session_id': session['id'], 'timestamp': session.get('timestamp', 0)}],
                        [embeddings]
                    )
                    
                    if session_graph is not None:
                        # Predict future states
                        classification, progression = self.tgn_model(
                            session_graph.x.to(self.device),
                            session_graph.edge_index.to(self.device),
                            session_graph.timestamps.to(self.device) if hasattr(session_graph, 'timestamps') else None
                        )
                        
                        trajectories.append({
                            'timestamp': session.get('timestamp', 0),
                            'embedding': embeddings.mean(dim=0).cpu(),
                            'predicted_next': progression.cpu(),
                            'classification': classification.cpu(),
                            'clinical_score': session.get('clinical_score', None),
                            'session_id': session['id']
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to analyze session {session.get('id', 'unknown')}: {e}")
                continue
        
        # Compute trajectory metrics
        metrics = self._compute_trajectory_metrics(trajectories)
        
        return trajectories, metrics
    
    def _compute_trajectory_metrics(self, trajectories: List[Dict]) -> Dict[str, float]:
        """
        Compute metrics characterizing gait progression.
        
        Args:
            trajectories: List of trajectory data points
            
        Returns:
            Dictionary of trajectory metrics
        """
        if len(trajectories) < 2:
            return {}
        
        metrics = {}
        
        # Recovery rate (change in embedding space)
        embeddings = [t['embedding'] for t in trajectories]
        distances = []
        
        for i in range(1, len(embeddings)):
            dist = torch.norm(embeddings[i] - embeddings[i-1])
            distances.append(dist.item())
        
        metrics['mean_change_rate'] = np.mean(distances)
        metrics['recovery_acceleration'] = np.diff(distances).mean() if len(distances) > 1 else 0.0
        
        # Trajectory smoothness
        metrics['smoothness'] = 1 / (np.std(distances) + 1e-6)
        
        # Distance to healthy baseline (if available)
        if hasattr(self, 'healthy_baseline'):
            final_dist = torch.norm(embeddings[-1] - self.healthy_baseline)
            initial_dist = torch.norm(embeddings[0] - self.healthy_baseline)
            metrics['recovery_percentage'] = (
                (initial_dist - final_dist) / initial_dist * 100
            )
        
        # Classification confidence trend
        if 'classification' in trajectories[0]:
            confidences = []
            for t in trajectories:
                conf = torch.softmax(t['classification'], dim=1).max().item()
                confidences.append(conf)
            
            metrics['confidence_trend'] = np.polyfit(range(len(confidences)), confidences, 1)[0]
            metrics['mean_confidence'] = np.mean(confidences)
        
        return metrics
    
    def predict_future_states(self, current_embeddings: torch.Tensor, 
                            num_steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future gait states.
        
        Args:
            current_embeddings: Current gait state embeddings
            num_steps: Number of future steps to predict
            
        Returns:
            Tuple of (predicted_states, confidence_scores)
        """
        # This is a simplified prediction - in practice, you might use a more sophisticated approach
        with torch.no_grad():
            # Use the progression head to predict next state
            # For multiple steps, we'd need to implement a more complex prediction mechanism
            
            # For now, return the current embeddings as placeholder
            predicted_states = current_embeddings.unsqueeze(0).repeat(num_steps, 1, 1)
            confidence_scores = torch.ones(num_steps, current_embeddings.shape[0])
            
            return predicted_states, confidence_scores


class ClinicalGaitAnalyzer:
    """
    Comprehensive clinical gait assessment system.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the clinical analyzer.
        
        Args:
            model_path: Path to trained model
            device: Device to use for inference
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load trained models
        self._load_models(model_path)
        
        # Load healthy baseline
        self.healthy_baseline = self._compute_healthy_baseline()
        
        logger.info(f"Initialized ClinicalGaitAnalyzer on device: {self.device}")
    
    def _load_models(self, model_path: str):
        """Load trained models from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize models
        self.encoder = GaitStateEncoder()
        self.tgn_model = TemporalGaitGNN()
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.tgn_model.load_state_dict(checkpoint['tgn_model_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.tgn_model.eval()
        
        # Move to device
        self.encoder.to(self.device)
        self.tgn_model.to(self.device)
    
    def _compute_healthy_baseline(self) -> torch.Tensor:
        """
        Compute average embedding for healthy controls.
        This is a placeholder - in practice, you would load actual healthy control data.
        """
        # For now, create a synthetic baseline
        baseline = torch.randn(128)  # 128 is the embedding dimension
        baseline = baseline / torch.norm(baseline)  # Normalize
        
        logger.info("Created synthetic healthy baseline (replace with actual data)")
        return baseline.to(self.device)
    
    def assess_patient(self, patient_grf_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive gait assessment for a patient.
        
        Args:
            patient_grf_data: Dictionary containing patient GRF data
            
        Returns:
            Comprehensive assessment report
        """
        report = {
            'timestamp': datetime.now(),
            'patient_id': patient_grf_data['id'],
            'assessments': []
        }
        
        # Process each session
        for session in patient_grf_data['sessions']:
            try:
                # Get embeddings
                grf_data = torch.tensor(session['grf_data'], dtype=torch.float).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.encoder(grf_data)
                
                # Build graph for this session
                from ..models import GaitGraphBuilder
                graph_builder = GaitGraphBuilder()
                
                session_graph = graph_builder.build_subject_graph(
                    [{'session_id': session['id'], 'timestamp': session.get('timestamp', 0)}],
                    [embeddings]
                )
                
                if session_graph is not None:
                    # Get predictions
                    classification, progression = self.tgn_model(
                        session_graph.x.to(self.device),
                        session_graph.edge_index.to(self.device),
                        session_graph.timestamps.to(self.device) if hasattr(session_graph, 'timestamps') else None
                    )
                    
                    # Compute risk scores
                    risk_score = self._compute_risk_score(embeddings, classification)
                    
                    # Asymmetry analysis
                    asymmetry = self._analyze_asymmetry(session['grf_data'])
                    
                    assessment = {
                        'session_id': session['id'],
                        'predicted_pathology': self._decode_pathology(classification),
                        'confidence': torch.softmax(classification, dim=1).max().item(),
                        'risk_score': risk_score,
                        'asymmetry_index': asymmetry,
                        'distance_from_healthy': torch.norm(
                            embeddings.mean(dim=0) - self.healthy_baseline
                        ).item(),
                        'embeddings': embeddings.cpu().numpy(),
                        'progression_prediction': progression.cpu().numpy()
                    }
                    
                    report['assessments'].append(assessment)
                    
            except Exception as e:
                logger.error(f"Failed to assess session {session.get('id', 'unknown')}: {e}")
                continue
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Overall risk assessment
        report['overall_risk'] = self._compute_overall_risk(report)
        
        return report
    
    def _compute_risk_score(self, embeddings: torch.Tensor, 
                           classification: torch.Tensor) -> float:
        """
        Compute composite risk score.
        
        Args:
            embeddings: Gait state embeddings
            classification: Pathology classification predictions
            
        Returns:
            Risk score between 0 and 1
        """
        # Probability of pathology
        probs = torch.softmax(classification, dim=1)
        pathology_prob = 1 - probs[0, 0]  # Not healthy control
        
        # Distance from healthy baseline
        distance = torch.norm(embeddings.mean(dim=0) - self.healthy_baseline)
        normalized_distance = torch.sigmoid(distance - 2)  # Threshold at 2
        
        # Combine scores
        risk_score = 0.6 * pathology_prob + 0.4 * normalized_distance
        
        return risk_score.item()
    
    def _analyze_asymmetry(self, grf_data: np.ndarray) -> float:
        """
        Analyze left-right asymmetry in gait.
        
        Args:
            grf_data: GRF data array
            
        Returns:
            Asymmetry index
        """
        try:
            # Assuming first two channels are left and right vertical forces
            if grf_data.shape[0] >= 2:
                left_forces = grf_data[0, :]
                right_forces = grf_data[1, :]
                
                # Compute asymmetry index
                left_mean = np.mean(left_forces)
                right_mean = np.mean(right_forces)
                
                if left_mean + right_mean > 0:
                    asymmetry = 2 * np.abs(left_mean - right_mean) / (left_mean + right_mean)
                else:
                    asymmetry = 0.0
                
                return asymmetry
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Failed to compute asymmetry: {e}")
            return 0.0
    
    def _decode_pathology(self, classification: torch.Tensor) -> str:
        """
        Convert model output to readable pathology.
        
        Args:
            classification: Model classification output
            
        Returns:
            Pathology name
        """
        pathology_names = ['Healthy', 'Hip', 'Knee', 'Ankle', 'Calcaneus']
        pred_idx = classification.argmax().item()
        return pathology_names[pred_idx]
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate clinical recommendations based on assessment.
        
        Args:
            report: Assessment report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not report['assessments']:
            return recommendations
        
        # Check latest assessment
        latest = report['assessments'][-1]
        
        # High risk recommendation
        if latest['risk_score'] > 0.7:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Immediate clinical evaluation recommended',
                'reason': f"High risk score ({latest['risk_score']:.2f})"
            })
        
        # Asymmetry recommendation
        if latest['asymmetry_index'] > 0.15:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Gait retraining focusing on symmetry',
                'reason': f"Significant asymmetry detected ({latest['asymmetry_index']:.2%})"
            })
        
        # Check progression if multiple assessments
        if len(report['assessments']) > 1:
            prev_risk = report['assessments'][-2]['risk_score']
            curr_risk = latest['risk_score']
            
            if curr_risk > prev_risk * 1.2:  # 20% increase
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Review treatment plan - deterioration detected',
                    'reason': 'Risk score increased by >20% since last session'
                })
            elif curr_risk < prev_risk * 0.8:  # 20% decrease
                recommendations.append({
                    'priority': 'LOW',
                    'action': 'Continue current treatment plan',
                    'reason': 'Positive response to treatment observed'
                })
        
        return recommendations
    
    def _compute_overall_risk(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall risk assessment.
        
        Args:
            report: Assessment report
            
        Returns:
            Overall risk summary
        """
        if not report['assessments']:
            return {'level': 'UNKNOWN', 'score': 0.0}
        
        # Get latest risk scores
        risk_scores = [a['risk_score'] for a in report['assessments']]
        latest_risk = risk_scores[-1]
        
        # Determine risk level
        if latest_risk > 0.7:
            risk_level = 'HIGH'
        elif latest_risk > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Trend analysis
        if len(risk_scores) > 1:
            trend = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
            if trend > 0.05:
                trend_direction = 'INCREASING'
            elif trend < -0.05:
                trend_direction = 'DECREASING'
            else:
                trend_direction = 'STABLE'
        else:
            trend_direction = 'UNKNOWN'
        
        return {
            'level': risk_level,
            'score': latest_risk,
            'trend': trend_direction,
            'mean_risk': np.mean(risk_scores),
            'risk_volatility': np.std(risk_scores)
        }
