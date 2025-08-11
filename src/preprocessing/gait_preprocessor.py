import pandas as pd
import numpy as np
import os
from pathlib import Path
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GaitDataPreprocessor:
    """
    Advanced gait data preprocessor for the GaitRec dataset.
    Handles GRF data loading, gait cycle detection, and normalization.
    """
    
    def __init__(self, sampling_rate: int = 2000, data_path: str = "data"):
        """
        Initialize the preprocessor.
        
        Args:
            sampling_rate: Sampling rate in Hz (GaitRec uses 2000 Hz)
            data_path: Path to the dataset directory
        """
        self.sampling_rate = sampling_rate
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        
        # Dataset structure from GaitRec
        self.data_files = {
            'vertical_force': ['GRF_F_V-RAW_LEFT.csv', 'GRF_F_V-RAW_RIGHT.csv'],
            'anterior_posterior': ['GRF_F_AP-RAW_LEFT.csv', 'GRF_F_AP-RAW_RIGHT.csv'],
            'medio_lateral': ['GRF_F_ML-RAW_LEFT.csv', 'GRF_F_ML-RAW_RIGHT.csv'],
            'cop_ap': ['GRF_COP_AP-RAW_LEFT.csv', 'GRF_COP_AP-RAW_RIGHT.csv'],
            'cop_ml': ['GRF_COP_ML-RAW_LEFT.csv', 'GRF_COP_ML-RAW_RIGHT.csv'],
            'metadata': 'GRF-metadata.csv'
        }
        
        # Verify data files exist
        self._verify_data_files()
    
    def _verify_data_files(self):
        """Verify that all required data files exist."""
        missing_files = []
        
        for file_list in self.data_files.values():
            if isinstance(file_list, list):
                for file in file_list:
                    if not (self.data_path / file).exists():
                        missing_files.append(file)
            else:
                if not (self.data_path / file_list).exists():
                    missing_files.append(file_list)
        
        if missing_files:
            logger.warning(f"Missing data files: {missing_files}")
            logger.warning("Please download the GaitRec dataset to the 'data' directory")
        else:
            logger.info("All data files found successfully")
    
    def load_metadata(self) -> pd.DataFrame:
        """Load the metadata file."""
        metadata_path = self.data_path / self.data_files['metadata']
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(metadata)} records")
        return metadata
    
    def load_trial(self, subject_id: str, session_id: str, side: str = 'both') -> Dict[str, np.ndarray]:
        """
        Load a single walking trial.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            side: 'left', 'right', or 'both'
            
        Returns:
            Dictionary containing trial data for all force components
        """
        trial_data = {}
        
        try:
            # Load all force components
            for force_type, file_names in self.data_files.items():
                if force_type == 'metadata':
                    continue
                    
                if side == 'both':
                    # Load both left and right data
                    left_file = self.data_path / file_names[0]
                    right_file = self.data_path / file_names[1]
                    
                    if left_file.exists() and right_file.exists():
                        left_data = pd.read_csv(left_file)
                        right_data = pd.read_csv(right_file)
                        
                        # Filter by subject and session
                        left_mask = (left_data['SUBJECT_ID'] == subject_id) & \
                                   (left_data['SESSION_ID'] == session_id)
                        right_mask = (right_data['SUBJECT_ID'] == subject_id) & \
                                    (right_data['SESSION_ID'] == session_id)
                        
                        if left_mask.any() and right_mask.any():
                            trial_data[f'{force_type}_L'] = left_data.loc[left_mask].values
                            trial_data[f'{force_type}_R'] = right_data.loc[right_mask].values
                        else:
                            logger.warning(f"No data found for subject {subject_id}, session {session_id}")
                            return {}
                    else:
                        logger.warning(f"Data files not found for {force_type}")
                        
                else:
                    # Load single side data
                    file_path = self.data_path / file_names[0] if 'LEFT' in file_names[0] else \
                               self.data_path / file_names[1]
                    
                    if file_path.exists():
                        data = pd.read_csv(file_path)
                        mask = (data['SUBJECT_ID'] == subject_id) & \
                               (data['SESSION_ID'] == session_id)
                        
                        if mask.any():
                            trial_data[force_type] = data.loc[mask].values
                        else:
                            logger.warning(f"No data found for subject {subject_id}, session {session_id}")
                            return {}
                    else:
                        logger.warning(f"Data file not found: {file_path}")
            
            if trial_data:
                logger.info(f"Successfully loaded trial for subject {subject_id}, session {session_id}")
            else:
                logger.warning(f"No trial data loaded for subject {subject_id}, session {session_id}")
                
        except Exception as e:
            logger.error(f"Error loading trial: {e}")
            return {}
            
        return trial_data
    
    def normalize_by_body_weight(self, forces: np.ndarray, body_mass: float) -> np.ndarray:
        """
        Normalize forces by body weight (N/kg).
        
        Args:
            forces: Force data array
            body_mass: Body mass in kg
            
        Returns:
            Normalized forces
        """
        if body_mass <= 0:
            logger.warning(f"Invalid body mass: {body_mass}, using default normalization")
            return forces
        
        body_weight = body_mass * 9.81  # Convert to Newtons
        return forces / body_weight
    
    def detect_gait_events(self, vertical_force: np.ndarray, threshold: float = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect heel strike and toe-off events from vertical force data.
        
        Args:
            vertical_force: Vertical force time series
            threshold: Force threshold for stance phase detection
            
        Returns:
            Tuple of (heel_strikes, toe_offs) indices
        """
        try:
            # Low-pass filter to remove noise
            b, a = signal.butter(4, 50, btype='low', fs=self.sampling_rate)
            filtered_force = signal.filtfilt(b, a, vertical_force)
            
            # Detect stance phase (force > threshold)
            stance_mask = filtered_force > threshold
            
            # Find heel strikes (start of stance)
            heel_strikes = np.where(np.diff(stance_mask.astype(int)) == 1)[0]
            
            # Find toe-offs (end of stance)
            toe_offs = np.where(np.diff(stance_mask.astype(int)) == -1)[0]
            
            # Ensure we have valid gait events
            if len(heel_strikes) < 2:
                logger.warning(f"Insufficient gait events detected: {len(heel_strikes)} heel strikes")
                return np.array([]), np.array([])
            
            logger.info(f"Detected {len(heel_strikes)} heel strikes and {len(toe_offs)} toe-offs")
            return heel_strikes, toe_offs
            
        except Exception as e:
            logger.error(f"Error detecting gait events: {e}")
            return np.array([]), np.array([])
    
    def segment_gait_cycles(self, trial_data: Dict[str, np.ndarray], 
                           metadata_row: pd.Series) -> List[Dict]:
        """
        Segment continuous data into individual gait cycles.
        
        Args:
            trial_data: Trial data dictionary
            metadata_row: Metadata row containing subject information
            
        Returns:
            List of gait cycle dictionaries
        """
        cycles = []
        
        try:
            # Check if we have bilateral data
            if 'vertical_force_L' not in trial_data or 'vertical_force_R' not in trial_data:
                logger.warning("Bilateral data not available for gait cycle segmentation")
                return cycles
            
            # Detect gait events from vertical force
            hs_left, to_left = self.detect_gait_events(trial_data['vertical_force_L'].flatten())
            hs_right, to_right = self.detect_gait_events(trial_data['vertical_force_R'].flatten())
            
            if len(hs_left) < 2 or len(hs_right) < 2:
                logger.warning("Insufficient gait events for cycle segmentation")
                return cycles
            
            # Extract individual cycles (heel strike to heel strike)
            for i in range(min(len(hs_left) - 1, len(hs_right) - 1)):
                # Left side cycle
                left_cycle = {
                    'start_idx': hs_left[i],
                    'end_idx': hs_left[i + 1],
                    'side': 'left',
                    'forces': {}
                }
                
                # Extract all force components for this cycle
                for force_type in ['vertical_force', 'anterior_posterior', 'medio_lateral', 'cop_ap', 'cop_ml']:
                    key = f'{force_type}_L'
                    if key in trial_data:
                        cycle_data = trial_data[key][left_cycle['start_idx']:left_cycle['end_idx']]
                        if len(cycle_data) > 0:
                            left_cycle['forces'][force_type] = cycle_data.flatten()
                
                # Normalize to 101 points (0-100% gait cycle)
                if left_cycle['forces']:
                    left_cycle['forces_normalized'] = self.normalize_cycle_length(
                        left_cycle['forces']
                    )
                    cycles.append(left_cycle)
                
                # Right side cycle
                right_cycle = {
                    'start_idx': hs_right[i],
                    'end_idx': hs_right[i + 1],
                    'side': 'right',
                    'forces': {}
                }
                
                # Extract all force components for this cycle
                for force_type in ['vertical_force', 'anterior_posterior', 'medio_lateral', 'cop_ap', 'cop_ml']:
                    key = f'{force_type}_R'
                    if key in trial_data:
                        cycle_data = trial_data[key][right_cycle['start_idx']:right_cycle['end_idx']]
                        if len(cycle_data) > 0:
                            right_cycle['forces'][force_type] = cycle_data.flatten()
                
                # Normalize to 101 points (0-100% gait cycle)
                if right_cycle['forces']:
                    right_cycle['forces_normalized'] = self.normalize_cycle_length(
                        right_cycle['forces']
                    )
                    cycles.append(right_cycle)
            
            logger.info(f"Successfully segmented {len(cycles)} gait cycles")
            
        except Exception as e:
            logger.error(f"Error segmenting gait cycles: {e}")
        
        return cycles
    
    def normalize_cycle_length(self, cycle_forces: Dict[str, np.ndarray], 
                              n_points: int = 101) -> Dict[str, np.ndarray]:
        """
        Normalize gait cycle to standard length.
        
        Args:
            cycle_forces: Dictionary of force data for one cycle
            n_points: Number of points in normalized cycle
            
        Returns:
            Dictionary of normalized force data
        """
        normalized = {}
        
        try:
            for force_type, force_data in cycle_forces.items():
                if len(force_data) > 1:
                    x_old = np.linspace(0, 100, len(force_data))
                    x_new = np.linspace(0, 100, n_points)
                    normalized[force_type] = np.interp(x_new, x_old, force_data)
                else:
                    # Handle single point data
                    normalized[force_type] = np.full(n_points, force_data[0] if len(force_data) > 0 else 0)
                    
        except Exception as e:
            logger.error(f"Error normalizing cycle length: {e}")
            # Return zeros if normalization fails
            for force_type in cycle_forces.keys():
                normalized[force_type] = np.zeros(n_points)
        
        return normalized
    
    def preprocess_subject(self, subject_id: str, metadata: pd.DataFrame) -> Dict:
        """
        Preprocess all sessions for a single subject.
        
        Args:
            subject_id: Subject identifier
            metadata: Metadata dataframe
            
        Returns:
            Dictionary containing preprocessed data for all sessions
        """
        subject_data = metadata[metadata['SUBJECT_ID'] == subject_id]
        
        if len(subject_data) == 0:
            logger.warning(f"No data found for subject {subject_id}")
            return {}
        
        processed_sessions = []
        
        for _, session_row in subject_data.iterrows():
            session_id = session_row['SESSION_ID']
            
            # Load trial data
            trial_data = self.load_trial(subject_id, session_id)
            
            if not trial_data:
                continue
            
            # Segment gait cycles
            cycles = self.segment_gait_cycles(trial_data, session_row)
            
            if cycles:
                processed_sessions.append({
                    'session_id': session_id,
                    'metadata': session_row.to_dict(),
                    'cycles': cycles,
                    'n_cycles': len(cycles)
                })
        
        logger.info(f"Preprocessed {len(processed_sessions)} sessions for subject {subject_id}")
        
        return {
            'subject_id': subject_id,
            'sessions': processed_sessions,
            'total_cycles': sum(s['n_cycles'] for s in processed_sessions)
        }
    
    def get_dataset_statistics(self) -> Dict:
        """Get basic statistics about the dataset."""
        try:
            metadata = self.load_metadata()
            
            stats = {
                'total_subjects': metadata['SUBJECT_ID'].nunique(),
                'total_sessions': len(metadata),
                'pathology_distribution': metadata['CLASS'].value_counts().to_dict(),
                'sex_distribution': metadata['SEX'].value_counts().to_dict(),
                'age_stats': {
                    'mean': metadata['AGE'].mean(),
                    'std': metadata['AGE'].std(),
                    'min': metadata['AGE'].min(),
                    'max': metadata['AGE'].max()
                },
                'body_mass_stats': {
                    'mean': metadata['BODY_MASS'].mean(),
                    'std': metadata['BODY_MASS'].std(),
                    'min': metadata['BODY_MASS'].min(),
                    'max': metadata['BODY_MASS'].max()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing dataset statistics: {e}")
            return {}
