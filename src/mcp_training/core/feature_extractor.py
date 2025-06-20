"""
WiFi feature extraction for anomaly detection training.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from .config import get_global_config

logger = logging.getLogger(__name__)


class WiFiFeatureExtractor:
    """Extract features from WiFi log data for anomaly detection."""
    
    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """Initialize the feature extractor."""
        config = get_global_config()
        self.feature_config = feature_config or config.get_feature_config()
        self.mac_pattern = re.compile(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})')
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        
    def extract_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract all features from the log data."""
        logger.info(f"Extracting features from {len(data)} log entries")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract different feature types
        features_df = pd.DataFrame()
        
        # Time-based features
        if self._is_feature_enabled('time_features'):
            time_features = self._extract_time_features(df)
            features_df = pd.concat([features_df, time_features], axis=1)
        
        # WiFi-specific features
        if self._is_feature_enabled('wifi_features'):
            wifi_features = self._extract_wifi_features(df)
            features_df = pd.concat([features_df, wifi_features], axis=1)
        
        # Text-based features
        if self._is_feature_enabled('text_features'):
            text_features = self._extract_text_features(df)
            features_df = pd.concat([features_df, text_features], axis=1)
        
        # Process-based features
        process_features = self._extract_process_features(df)
        features_df = pd.concat([features_df, process_features], axis=1)
        
        # Aggregate features over time windows
        window_features = self._extract_window_features(df)
        features_df = pd.concat([features_df, window_features], axis=1)
        
        logger.info(f"Extracted {features_df.shape[1]} features")
        return features_df
    
    def _is_feature_enabled(self, feature_type: str) -> bool:
        """Check if a feature type is enabled in configuration."""
        features = self.feature_config.get(feature_type, {})
        return features.get('enabled', True)
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        features = {}
        
        # Hour of day (0-23)
        features['hour_of_day'] = df['timestamp'].dt.hour
        
        # Day of week (0-6, Monday=0)
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Minute of hour (0-59)
        features['minute_of_hour'] = df['timestamp'].dt.minute
        
        # Time since midnight (seconds)
        features['time_since_midnight'] = (
            df['timestamp'].dt.hour * 3600 + 
            df['timestamp'].dt.minute * 60 + 
            df['timestamp'].dt.second
        )
        
        # Weekend indicator
        features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Business hours indicator (9 AM - 5 PM)
        features['is_business_hours'] = (
            (df['timestamp'].dt.hour >= 9) & 
            (df['timestamp'].dt.hour < 17)
        ).astype(int)
        
        return pd.DataFrame(features)
    
    def _extract_wifi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract WiFi-specific features."""
        features = {}
        
        # Extract MAC addresses from messages
        df['mac_addresses'] = df['message'].apply(self._extract_mac_addresses)
        df['ip_addresses'] = df['message'].apply(self._extract_ip_addresses)
        
        # Connection events
        features['is_connection_event'] = df['message'].str.contains(
            'AP-STA-CONNECTED|AP-STA-DISCONNECTED|STA-ASSOC|STA-DISASSOC',
            case=False, regex=True
        ).astype(int)
        
        # Authentication events
        features['is_auth_event'] = df['message'].str.contains(
            'AUTH|AUTHENTICATION|LOGIN|LOGOUT',
            case=False, regex=True
        ).astype(int)
        
        # Error events
        features['is_error_event'] = df['message'].str.contains(
            'ERROR|FAILED|DENIED|REJECTED',
            case=False, regex=True
        ).astype(int)
        
        # MAC address count in message
        features['mac_count'] = df['mac_addresses'].apply(len)
        
        # IP address count in message
        features['ip_count'] = df['ip_addresses'].apply(len)
        
        # Message length
        features['message_length'] = df['message'].str.len()
        
        return pd.DataFrame(features)
    
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text-based features."""
        features = {}
        
        # Word count
        features['word_count'] = df['message'].str.split().str.len()
        
        # Special character count
        features['special_char_count'] = df['message'].str.count(r'[^a-zA-Z0-9\s]')
        
        # Uppercase ratio
        features['uppercase_ratio'] = (
            df['message'].str.count(r'[A-Z]') / 
            df['message'].str.len().replace(0, 1)
        )
        
        # Number count
        features['number_count'] = df['message'].str.count(r'\d')
        
        # Unique characters
        features['unique_chars'] = df['message'].apply(lambda x: len(set(x)))
        
        return pd.DataFrame(features)
    
    def _extract_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract process-based features."""
        features = {}
        
        # Process name encoding
        process_counts = df['process_name'].value_counts()
        features['process_rank'] = df['process_name'].map(process_counts.rank(method='dense'))
        
        # Log level encoding
        log_levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        features['log_level_numeric'] = df['log_level'].map(log_levels).fillna(1)
        
        # Process activity frequency
        process_freq = df['process_name'].value_counts(normalize=True)
        features['process_frequency'] = df['process_name'].map(process_freq)
        
        return pd.DataFrame(features)
    
    def _extract_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features over time windows."""
        features = {}
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # 5-minute window features
        window_5min = self._calculate_window_features(df_sorted, window_minutes=5)
        features.update({f'window_5min_{k}': v for k, v in window_5min.items()})
        
        # 15-minute window features
        window_15min = self._calculate_window_features(df_sorted, window_minutes=15)
        features.update({f'window_15min_{k}': v for k, v in window_15min.items()})
        
        # 1-hour window features
        window_1hour = self._calculate_window_features(df_sorted, window_minutes=60)
        features.update({f'window_1hour_{k}': v for k, v in window_1hour.items()})
        
        return pd.DataFrame(features)
    
    def _calculate_window_features(self, df: pd.DataFrame, window_minutes: int) -> Dict[str, pd.Series]:
        """Calculate features over a sliding time window."""
        features = {}
        window = timedelta(minutes=window_minutes)
        
        # Connection count in window
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        features['connection_count'] = self._rolling_count(df, connection_mask, window)
        
        # Unique MAC addresses in window
        features['unique_macs'] = self._rolling_unique_count(df, 'mac_addresses', window)
        
        # Error count in window
        error_mask = df['message'].str.contains('ERROR|FAILED', case=False)
        features['error_count'] = self._rolling_count(df, error_mask, window)
        
        # Process diversity in window
        features['process_diversity'] = self._rolling_unique_count(df, 'process_name', window)
        
        return features
    
    def _rolling_count(self, df: pd.DataFrame, mask: pd.Series, window: timedelta) -> pd.Series:
        """Count events in rolling time window."""
        counts = []
        for i, timestamp in enumerate(df['timestamp']):
            window_start = timestamp - window
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= timestamp)
            counts.append(mask[window_mask].sum())
        return pd.Series(counts, index=df.index)
    
    def _rolling_unique_count(self, df: pd.DataFrame, column: str, window: timedelta) -> pd.Series:
        """Count unique values in rolling time window."""
        counts = []
        for i, timestamp in enumerate(df['timestamp']):
            window_start = timestamp - window
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= timestamp)
            if column == 'mac_addresses':
                # Flatten list of MAC addresses
                all_macs = []
                for macs in df.loc[window_mask, column]:
                    all_macs.extend(macs)
                counts.append(len(set(all_macs)))
            else:
                counts.append(df.loc[window_mask, column].nunique())
        return pd.Series(counts, index=df.index)
    
    def _extract_mac_addresses(self, text: str) -> List[str]:
        """Extract MAC addresses from text."""
        return self.mac_pattern.findall(text)
    
    def _extract_ip_addresses(self, text: str) -> List[str]:
        """Extract IP addresses from text."""
        return self.ip_pattern.findall(text)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # This would be populated after feature extraction
        # For now, return expected feature names
        return [
            'hour_of_day', 'day_of_week', 'minute_of_hour', 'time_since_midnight',
            'is_weekend', 'is_business_hours', 'is_connection_event', 'is_auth_event',
            'is_error_event', 'mac_count', 'ip_count', 'message_length', 'word_count',
            'special_char_count', 'uppercase_ratio', 'number_count', 'unique_chars',
            'process_rank', 'log_level_numeric', 'process_frequency'
        ] 