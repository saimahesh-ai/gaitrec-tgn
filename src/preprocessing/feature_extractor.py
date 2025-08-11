import numpy as np
import scipy.signal as signal
from scipy import stats
from scipy.fft import fft
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GaitFeatureExtractor:
    """
    Comprehensive feature extractor for gait analysis.
    Extracts temporal, frequency, symmetry, and COP features from GRF data.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
        
    def extract_temporal_features(self, forces: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from force data.
        
        Args:
            forces: Force time series data
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        try:
            if len(forces) == 0:
                logger.warning("Empty force data for temporal feature extraction")
                return self._get_empty_temporal_features()
            
            # Statistical features
            features['mean'] = float(np.mean(forces))
            features['std'] = float(np.std(forces))
            features['max'] = float(np.max(forces))
            features['min'] = float(np.min(forces))
            features['range'] = float(features['max'] - features['min'])
            
            # Higher-order moments
            if len(forces) > 3:
                features['skewness'] = float(stats.skew(forces))
                features['kurtosis'] = float(stats.kurtosis(forces))
            else:
                features['skewness'] = 0.0
                features['kurtosis'] = 0.0
            
            # Peak characteristics
            try:
                peaks, properties = signal.find_peaks(forces, height=0)
                if len(peaks) > 0:
                    features['n_peaks'] = len(peaks)
                    features['max_peak_height'] = float(np.max(properties['peak_heights']))
                    features['mean_peak_height'] = float(np.mean(properties['peak_heights']))
                    features['peak_std'] = float(np.std(properties['peak_heights']))
                else:
                    features['n_peaks'] = 0
                    features['max_peak_height'] = 0.0
                    features['mean_peak_height'] = 0.0
                    features['peak_std'] = 0.0
            except Exception as e:
                logger.warning(f"Error in peak detection: {e}")
                features['n_peaks'] = 0
                features['max_peak_height'] = 0.0
                features['mean_peak_height'] = 0.0
                features['peak_std'] = 0.0
            
            # Loading rate (max derivative)
            if len(forces) > 1:
                derivative = np.diff(forces)
                features['max_loading_rate'] = float(np.max(np.abs(derivative)))
                features['mean_loading_rate'] = float(np.mean(np.abs(derivative)))
                features['loading_rate_std'] = float(np.std(np.abs(derivative)))
            else:
                features['max_loading_rate'] = 0.0
                features['mean_loading_rate'] = 0.0
                features['loading_rate_std'] = 0.0
            
            # Area under curve
            features['auc'] = float(np.trapz(forces))
            
            # Root mean square
            features['rms'] = float(np.sqrt(np.mean(forces**2)))
            
            # Coefficient of variation
            if features['mean'] != 0:
                features['cv'] = float(features['std'] / abs(features['mean']))
            else:
                features['cv'] = 0.0
                
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return self._get_empty_temporal_features()
        
        return features
    
    def _get_empty_temporal_features(self) -> Dict[str, float]:
        """Return empty temporal features when data is unavailable."""
        return {
            'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0, 'range': 0.0,
            'skewness': 0.0, 'kurtosis': 0.0, 'n_peaks': 0, 'max_peak_height': 0.0,
            'mean_peak_height': 0.0, 'peak_std': 0.0, 'max_loading_rate': 0.0,
            'mean_loading_rate': 0.0, 'loading_rate_std': 0.0, 'auc': 0.0,
            'rms': 0.0, 'cv': 0.0
        }
    
    def extract_frequency_features(self, forces: np.ndarray, 
                                 sampling_rate: float = 100.0) -> Dict[str, float]:
        """
        Extract frequency-domain features from force data.
        
        Args:
            forces: Force time series data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        try:
            if len(forces) < 4:
                logger.warning("Insufficient data for frequency feature extraction")
                return self._get_empty_frequency_features()
            
            # FFT
            fft_vals = fft(forces)
            fft_freq = np.fft.fftfreq(len(forces), 1/sampling_rate)
            
            # Power spectrum
            power = np.abs(fft_vals) ** 2
            
            # Use only positive frequencies
            positive_freq_mask = fft_freq > 0
            positive_freqs = fft_freq[positive_freq_mask]
            positive_power = power[positive_freq_mask]
            
            if len(positive_power) == 0:
                return self._get_empty_frequency_features()
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(positive_power)
            features['dominant_freq'] = float(positive_freqs[dominant_freq_idx])
            
            # Spectral entropy
            psd_norm = positive_power / np.sum(positive_power)
            features['spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
            
            # Spectral centroid
            features['spectral_centroid'] = float(np.sum(positive_freqs * positive_power) / np.sum(positive_power))
            
            # Spectral bandwidth
            centroid = features['spectral_centroid']
            features['spectral_bandwidth'] = float(np.sqrt(np.sum(positive_power * (positive_freqs - centroid)**2) / np.sum(positive_power)))
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_power = np.cumsum(positive_power)
            rolloff_idx = np.where(cumulative_power >= 0.85 * cumulative_power[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = float(positive_freqs[rolloff_idx[0]])
            else:
                features['spectral_rolloff'] = 0.0
            
            # Zero crossing rate
            features['zero_crossing_rate'] = float(np.sum(np.diff(np.sign(forces)) != 0) / (len(forces) - 1))
            
        except Exception as e:
            logger.error(f"Error extracting frequency features: {e}")
            return self._get_empty_frequency_features()
        
        return features
    
    def _get_empty_frequency_features(self) -> Dict[str, float]:
        """Return empty frequency features when data is unavailable."""
        return {
            'dominant_freq': 0.0, 'spectral_entropy': 0.0, 'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0, 'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0
        }
    
    def extract_symmetry_features(self, left_forces: np.ndarray, 
                                 right_forces: np.ndarray) -> Dict[str, float]:
        """
        Extract inter-limb symmetry features.
        
        Args:
            left_forces: Left limb force data
            right_forces: Right limb force data
            
        Returns:
            Dictionary of symmetry features
        """
        features = {}
        
        try:
            if len(left_forces) == 0 or len(right_forces) == 0:
                logger.warning("Empty force data for symmetry feature extraction")
                return self._get_empty_symmetry_features()
            
            # Ensure same length for comparison
            min_length = min(len(left_forces), len(right_forces))
            left_trimmed = left_forces[:min_length]
            right_trimmed = right_forces[:min_length]
            
            # Symmetry index (SI)
            left_mean = np.mean(left_trimmed)
            right_mean = np.mean(right_trimmed)
            
            if left_mean + right_mean != 0:
                features['symmetry_index'] = float(2 * np.abs(left_mean - right_mean) / (left_mean + right_mean))
            else:
                features['symmetry_index'] = 0.0
            
            # Cross-correlation
            try:
                correlation = np.correlate(left_trimmed, right_trimmed, mode='same')
                features['max_correlation'] = float(np.max(correlation))
                features['min_correlation'] = float(np.min(correlation))
                features['mean_correlation'] = float(np.mean(correlation))
            except Exception as e:
                logger.warning(f"Error computing correlation: {e}")
                features['max_correlation'] = 0.0
                features['min_correlation'] = 0.0
                features['mean_correlation'] = 0.0
            
            # Phase shift
            try:
                lag = np.argmax(correlation) - len(correlation) // 2
                features['phase_shift'] = float(lag)
            except Exception as e:
                logger.warning(f"Error computing phase shift: {e}")
                features['phase_shift'] = 0.0
            
            # Asymmetry ratio
            if right_mean != 0:
                features['asymmetry_ratio'] = float(left_mean / right_mean)
            else:
                features['asymmetry_ratio'] = 0.0
            
            # Coefficient of variation of asymmetry
            asymmetry_values = np.abs(left_trimmed - right_trimmed)
            if np.mean(asymmetry_values) != 0:
                features['asymmetry_cv'] = float(np.std(asymmetry_values) / np.mean(asymmetry_values))
            else:
                features['asymmetry_cv'] = 0.0
                
        except Exception as e:
            logger.error(f"Error extracting symmetry features: {e}")
            return self._get_empty_symmetry_features()
        
        return features
    
    def _get_empty_symmetry_features(self) -> Dict[str, float]:
        """Return empty symmetry features when data is unavailable."""
        return {
            'symmetry_index': 0.0, 'max_correlation': 0.0, 'min_correlation': 0.0,
            'mean_correlation': 0.0, 'phase_shift': 0.0, 'asymmetry_ratio': 0.0,
            'asymmetry_cv': 0.0
        }
    
    def extract_cop_features(self, cop_ap: np.ndarray, cop_ml: np.ndarray) -> Dict[str, float]:
        """
        Extract center of pressure trajectory features.
        
        Args:
            cop_ap: Anterior-posterior COP coordinates
            cop_ml: Medio-lateral COP coordinates
            
        Returns:
            Dictionary of COP features
        """
        features = {}
        
        try:
            if len(cop_ap) == 0 or len(cop_ml) == 0:
                logger.warning("Empty COP data for feature extraction")
                return self._get_empty_cop_features()
            
            # Ensure same length
            min_length = min(len(cop_ap), len(cop_ml))
            cop_ap_trimmed = cop_ap[:min_length]
            cop_ml_trimmed = cop_ml[:min_length]
            
            # Path length
            if min_length > 1:
                path_length = np.sum(np.sqrt(np.diff(cop_ap_trimmed)**2 + np.diff(cop_ml_trimmed)**2))
                features['cop_path_length'] = float(path_length)
            else:
                features['cop_path_length'] = 0.0
            
            # Sway area (convex hull)
            try:
                if min_length >= 3:
                    points = np.column_stack((cop_ap_trimmed, cop_ml_trimmed))
                    unique_points = np.unique(points, axis=0)
                    
                    if len(unique_points) >= 3:
                        hull = ConvexHull(unique_points)
                        features['cop_sway_area'] = float(hull.volume)  # 2D area
                    else:
                        features['cop_sway_area'] = 0.0
                else:
                    features['cop_sway_area'] = 0.0
            except Exception as e:
                logger.warning(f"Error computing convex hull: {e}")
                features['cop_sway_area'] = 0.0
            
            # Velocity and acceleration
            if min_length > 1:
                cop_velocity = np.sqrt(np.diff(cop_ap_trimmed)**2 + np.diff(cop_ml_trimmed)**2)
                features['cop_mean_velocity'] = float(np.mean(cop_velocity))
                features['cop_max_velocity'] = float(np.max(cop_velocity))
                features['cop_velocity_std'] = float(np.std(cop_velocity))
                
                if min_length > 2:
                    cop_acceleration = np.diff(cop_velocity)
                    features['cop_mean_acceleration'] = float(np.mean(cop_acceleration))
                    features['cop_max_acceleration'] = float(np.max(cop_acceleration))
                    features['cop_acceleration_std'] = float(np.std(cop_acceleration))
                else:
                    features['cop_mean_acceleration'] = 0.0
                    features['cop_max_acceleration'] = 0.0
                    features['cop_acceleration_std'] = 0.0
            else:
                features['cop_mean_velocity'] = 0.0
                features['cop_max_velocity'] = 0.0
                features['cop_velocity_std'] = 0.0
                features['cop_mean_acceleration'] = 0.0
                features['cop_max_acceleration'] = 0.0
                features['cop_acceleration_std'] = 0.0
            
            # COP range
            features['cop_ap_range'] = float(np.max(cop_ap_trimmed) - np.min(cop_ap_trimmed))
            features['cop_ml_range'] = float(np.max(cop_ml_trimmed) - np.min(cop_ml_trimmed))
            
            # COP center of mass
            features['cop_ap_com'] = float(np.mean(cop_ap_trimmed))
            features['cop_ml_com'] = float(np.mean(cop_ml_trimmed))
            
        except Exception as e:
            logger.error(f"Error extracting COP features: {e}")
            return self._get_empty_cop_features()
        
        return features
    
    def _get_empty_cop_features(self) -> Dict[str, float]:
        """Return empty COP features when data is unavailable."""
        return {
            'cop_path_length': 0.0, 'cop_sway_area': 0.0, 'cop_mean_velocity': 0.0,
            'cop_max_velocity': 0.0, 'cop_velocity_std': 0.0, 'cop_mean_acceleration': 0.0,
            'cop_max_acceleration': 0.0, 'cop_acceleration_std': 0.0, 'cop_ap_range': 0.0,
            'cop_ml_range': 0.0, 'cop_ap_com': 0.0, 'cop_ml_com': 0.0
        }
    
    def extract_all_features(self, gait_cycle: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract all features from a gait cycle.
        
        Args:
            gait_cycle: Dictionary containing normalized force data for one cycle
            
        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}
        
        try:
            # Extract features for each force component
            for force_type, force_data in gait_cycle['forces_normalized'].items():
                if len(force_data) > 0:
                    # Temporal features
                    temp_features = self.extract_temporal_features(force_data)
                    for key, value in temp_features.items():
                        all_features[f'{force_type}_{key}'] = value
                    
                    # Frequency features
                    freq_features = self.extract_frequency_features(force_data)
                    for key, value in freq_features.items():
                        all_features[f'{force_type}_{key}'] = value
            
            # Extract symmetry features if bilateral data available
            if 'vertical_force_L' in gait_cycle['forces_normalized'] and 'vertical_force_R' in gait_cycle['forces_normalized']:
                left_vf = gait_cycle['forces_normalized']['vertical_force_L']
                right_vf = gait_cycle['forces_normalized']['vertical_force_R']
                
                if len(left_vf) > 0 and len(right_vf) > 0:
                    sym_features = self.extract_symmetry_features(left_vf, right_vf)
                    for key, value in sym_features.items():
                        all_features[f'symmetry_{key}'] = value
            
            # Extract COP features if available
            if 'cop_ap_L' in gait_cycle['forces_normalized'] and 'cop_ml_L' in gait_cycle['forces_normalized']:
                cop_ap_l = gait_cycle['forces_normalized']['cop_ap_L']
                cop_ml_l = gait_cycle['forces_normalized']['cop_ml_L']
                
                if len(cop_ap_l) > 0 and len(cop_ml_l) > 0:
                    cop_features = self.extract_cop_features(cop_ap_l, cop_ml_l)
                    for key, value in cop_features.items():
                        all_features[f'cop_L_{key}'] = value
            
            if 'cop_ap_R' in gait_cycle['forces_normalized'] and 'cop_ml_R' in gait_cycle['forces_normalized']:
                cop_ap_r = gait_cycle['forces_normalized']['cop_ap_R']
                cop_ml_r = gait_cycle['forces_normalized']['cop_ml_R']
                
                if len(cop_ap_r) > 0 and len(cop_ml_r) > 0:
                    cop_features = self.extract_cop_features(cop_ap_r, cop_ml_r)
                    for key, value in cop_features.items():
                        all_features[f'cop_R_{key}'] = value
            
            # Add cycle metadata
            all_features['cycle_side'] = 1.0 if gait_cycle['side'] == 'right' else 0.0
            all_features['cycle_length'] = float(gait_cycle['end_idx'] - gait_cycle['start_idx'])
            
            logger.info(f"Extracted {len(all_features)} features from gait cycle")
            
        except Exception as e:
            logger.error(f"Error extracting all features: {e}")
            all_features = {}
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if not self.feature_names:
            # Generate feature names based on expected structure
            force_types = ['vertical_force', 'anterior_posterior', 'medio_lateral', 'cop_ap', 'cop_ml']
            sides = ['L', 'R']
            
            # Temporal features
            temp_features = ['mean', 'std', 'max', 'min', 'range', 'skewness', 'kurtosis',
                           'n_peaks', 'max_peak_height', 'mean_peak_height', 'peak_std',
                           'max_loading_rate', 'mean_loading_rate', 'loading_rate_std',
                           'auc', 'rms', 'cv']
            
            # Frequency features
            freq_features = ['dominant_freq', 'spectral_entropy', 'spectral_centroid',
                           'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']
            
            # Symmetry features
            sym_features = ['symmetry_index', 'max_correlation', 'min_correlation',
                          'mean_correlation', 'phase_shift', 'asymmetry_ratio', 'asymmetry_cv']
            
            # COP features
            cop_features = ['cop_path_length', 'cop_sway_area', 'cop_mean_velocity',
                          'cop_max_velocity', 'cop_velocity_std', 'cop_mean_acceleration',
                          'cop_max_acceleration', 'cop_acceleration_std', 'cop_ap_range',
                          'cop_ml_range', 'cop_ap_com', 'cop_ml_com']
            
            # Generate all feature names
            for force_type in force_types:
                for side in sides:
                    for feature in temp_features + freq_features:
                        self.feature_names.append(f'{force_type}_{side}_{feature}')
            
            # Add symmetry features
            for feature in sym_features:
                self.feature_names.append(f'symmetry_{feature}')
            
            # Add COP features
            for side in sides:
                for feature in cop_features:
                    self.feature_names.append(f'cop_{side}_{feature}')
            
            # Add cycle metadata
            self.feature_names.extend(['cycle_side', 'cycle_length'])
        
        return self.feature_names
