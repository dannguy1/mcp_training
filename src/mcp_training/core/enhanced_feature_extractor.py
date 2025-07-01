"""
Enhanced WiFi feature extraction for anomaly detection training.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnhancedWiFiFeatureExtractor:
    """Enhanced feature extractor for WiFi anomaly detection with advanced features."""
    
    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced feature extractor."""
        self.feature_config = feature_config or self._get_default_config()
        self.mac_pattern = re.compile(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})')
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.ssid_pattern = re.compile(r'SSID[:\s]+([^\s]+)', re.IGNORECASE)
        self.channel_pattern = re.compile(r'channel[:\s]+(\d+)', re.IGNORECASE)
        self.signal_pattern = re.compile(r'signal[:\s]*(-?\d+)', re.IGNORECASE)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature configuration."""
        return {
            'time_features': {'enabled': True, 'include_cyclical': True},
            'wifi_features': {'enabled': True, 'include_signal': True, 'include_channel': True},
            'behavioral_features': {'enabled': True, 'include_patterns': True},
            'network_features': {'enabled': True, 'include_topology': True},
            'statistical_features': {'enabled': True, 'include_distributions': True},
            'text_features': {'enabled': True, 'include_semantic': True},
            'window_features': {'enabled': True, 'windows': [5, 15, 60, 240]},  # 5min, 15min, 1hr, 4hr
            'advanced_features': {'enabled': True, 'include_anomaly_scores': True}
        }
    
    def extract_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract comprehensive features from WiFi log data."""
        logger.info(f"Extracting enhanced features from {len(data)} log entries")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp for time-based features
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract different feature types
        features_df = pd.DataFrame()
        
        # Basic WiFi features
        if self._is_feature_enabled('wifi_features'):
            wifi_features = self._extract_enhanced_wifi_features(df)
            features_df = pd.concat([features_df, wifi_features], axis=1)
        
        # Time-based features with cyclical encoding
        if self._is_feature_enabled('time_features'):
            time_features = self._extract_enhanced_time_features(df)
            features_df = pd.concat([features_df, time_features], axis=1)
        
        # Behavioral features
        if self._is_feature_enabled('behavioral_features'):
            behavioral_features = self._extract_behavioral_features(df)
            features_df = pd.concat([features_df, behavioral_features], axis=1)
        
        # Network topology features
        if self._is_feature_enabled('network_features'):
            network_features = self._extract_network_features(df)
            features_df = pd.concat([features_df, network_features], axis=1)
        
        # Statistical features
        if self._is_feature_enabled('statistical_features'):
            statistical_features = self._extract_statistical_features(df)
            features_df = pd.concat([features_df, statistical_features], axis=1)
        
        # Enhanced text features
        if self._is_feature_enabled('text_features'):
            text_features = self._extract_enhanced_text_features(df)
            features_df = pd.concat([features_df, text_features], axis=1)
        
        # Advanced window features
        if self._is_feature_enabled('window_features'):
            window_features = self._extract_advanced_window_features(df)
            features_df = pd.concat([features_df, window_features], axis=1)
        
        # Advanced anomaly detection features
        if self._is_feature_enabled('advanced_features'):
            advanced_features = self._extract_advanced_features(df)
            features_df = pd.concat([features_df, advanced_features], axis=1)
        
        logger.info(f"Extracted {features_df.shape[1]} enhanced features")
        return features_df
    
    def _is_feature_enabled(self, feature_type: str) -> bool:
        """Check if a feature type is enabled in configuration."""
        features = self.feature_config.get(feature_type, {})
        return features.get('enabled', True)
    
    def _extract_enhanced_wifi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced WiFi-specific features."""
        features = {}
        
        # Extract network information
        df['mac_addresses'] = df['message'].apply(self._extract_mac_addresses)
        df['ip_addresses'] = df['message'].apply(self._extract_ip_addresses)
        df['ssid'] = df['message'].apply(self._extract_ssid)
        df['channel'] = df['message'].apply(self._extract_channel)
        df['signal_strength'] = df['message'].apply(self._extract_signal_strength)
        
        # Connection events with more granularity
        features['is_connection_event'] = df['message'].str.contains(
            'AP-STA-CONNECTED|AP-STA-DISCONNECTED|STA-ASSOC|STA-DISASSOC',
            case=False, regex=True
        ).astype(int)
        
        features['is_connect_event'] = df['message'].str.contains(
            'AP-STA-CONNECTED|STA-ASSOC',
            case=False, regex=True
        ).astype(int)
        
        features['is_disconnect_event'] = df['message'].str.contains(
            'AP-STA-DISCONNECTED|STA-DISASSOC',
            case=False, regex=True
        ).astype(int)
        
        # Authentication events
        features['is_auth_event'] = df['message'].str.contains(
            'AUTH|AUTHENTICATION|LOGIN|LOGOUT',
            case=False, regex=True
        ).astype(int)
        
        # Security events
        features['is_security_event'] = df['message'].str.contains(
            'WPA|WEP|RSN|ENCRYPTION|SECURITY',
            case=False, regex=True
        ).astype(int)
        
        # Error events with severity
        features['is_error_event'] = df['message'].str.contains(
            'ERROR|FAILED|DENIED|REJECTED',
            case=False, regex=True
        ).astype(int)
        
        features['is_critical_error'] = df['message'].str.contains(
            'CRITICAL|FATAL|EMERGENCY',
            case=False, regex=True
        ).astype(int)
        
        # Network information
        features['mac_count'] = df['mac_addresses'].apply(len)
        features['ip_count'] = df['ip_addresses'].apply(len)
        features['has_ssid'] = df['ssid'].notna().astype(int)
        features['has_channel'] = df['channel'].notna().astype(int)
        features['has_signal'] = df['signal_strength'].notna().astype(int)
        
        # Signal strength features (if available)
        if self.feature_config['wifi_features'].get('include_signal', True):
            features['signal_strength_numeric'] = pd.to_numeric(df['signal_strength'], errors='coerce').fillna(-100)
            features['signal_strength_bin'] = pd.cut(
                features['signal_strength_numeric'], 
                bins=[-100, -80, -60, -40, 0], 
                labels=[0, 1, 2, 3], 
                include_lowest=True
            ).astype(float)
        
        # Channel features (if available)
        if self.feature_config['wifi_features'].get('include_channel', True):
            features['channel_numeric'] = pd.to_numeric(df['channel'], errors='coerce').fillna(0)
            features['is_2_4ghz'] = ((features['channel_numeric'] >= 1) & (features['channel_numeric'] <= 14)).astype(int)
            features['is_5ghz'] = ((features['channel_numeric'] >= 36) & (features['channel_numeric'] <= 165)).astype(int)
        
        # Message characteristics
        features['message_length'] = df['message'].str.len()
        features['message_complexity'] = df['message'].apply(self._calculate_message_complexity)
        
        return pd.DataFrame(features)
    
    def _extract_enhanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced time-based features with cyclical encoding."""
        features = {}
        
        # Basic time features
        features['hour_of_day'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['minute_of_hour'] = df['timestamp'].dt.minute
        features['second_of_minute'] = df['timestamp'].dt.second
        
        # Cyclical encoding for periodic features
        if self.feature_config['time_features'].get('include_cyclical', True):
            # Hour cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
            
            # Day of week cyclical encoding
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            # Minute cyclical encoding
            features['minute_sin'] = np.sin(2 * np.pi * features['minute_of_hour'] / 60)
            features['minute_cos'] = np.cos(2 * np.pi * features['minute_of_hour'] / 60)
        
        # Time since reference points
        features['time_since_midnight'] = (
            df['timestamp'].dt.hour * 3600 + 
            df['timestamp'].dt.minute * 60 + 
            df['timestamp'].dt.second
        )
        
        features['time_since_week_start'] = (
            df['timestamp'].dt.dayofweek * 24 * 3600 + 
            features['time_since_midnight']
        )
        
        # Time-based indicators
        features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        features['is_business_hours'] = (
            (df['timestamp'].dt.hour >= 9) & 
            (df['timestamp'].dt.hour < 17) &
            (df['timestamp'].dt.dayofweek < 5)
        ).astype(int)
        
        features['is_night_hours'] = (
            (df['timestamp'].dt.hour >= 22) | 
            (df['timestamp'].dt.hour < 6)
        ).astype(int)
        
        features['is_peak_hours'] = (
            ((df['timestamp'].dt.hour >= 7) & (df['timestamp'].dt.hour <= 9)) |
            ((df['timestamp'].dt.hour >= 17) & (df['timestamp'].dt.hour <= 19))
        ).astype(int)
        
        return pd.DataFrame(features)
    
    def _extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral patterns and anomalies."""
        features = {}
        
        # Connection patterns
        features['connection_rate'] = self._calculate_connection_rate(df)
        features['disconnection_rate'] = self._calculate_disconnection_rate(df)
        features['connection_ratio'] = features['connection_rate'] / (features['disconnection_rate'] + 1e-8)
        
        # Burst detection
        features['connection_burst'] = self._detect_burst_activity(df, 'AP-STA-CONNECTED')
        features['error_burst'] = self._detect_burst_activity(df, 'ERROR|FAILED')
        
        # Device behavior
        features['unique_devices_per_hour'] = self._calculate_unique_devices_per_window(df, hours=1)
        features['device_connection_frequency'] = self._calculate_device_connection_frequency(df)
        
        # Temporal patterns
        features['activity_regularity'] = self._calculate_activity_regularity(df)
        features['peak_activity_hour'] = self._find_peak_activity_hour(df)
        
        # Anomalous patterns
        features['unusual_connection_time'] = self._detect_unusual_connection_times(df)
        features['rapid_reconnections'] = self._detect_rapid_reconnections(df)
        
        return pd.DataFrame(features)
    
    def _extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract network topology and infrastructure features."""
        features = {}
        
        # Network topology
        features['network_density'] = self._calculate_network_density(df)
        features['network_centralization'] = self._calculate_network_centralization(df)
        features['network_clustering'] = self._calculate_network_clustering(df)
        
        # Channel utilization
        features['channel_utilization'] = self._calculate_channel_utilization(df)
        features['channel_interference'] = self._calculate_channel_interference(df)
        features['channel_switching_frequency'] = self._calculate_channel_switching_frequency(df)
        
        # SSID analysis
        features['ssid_diversity'] = self._calculate_ssid_diversity(df)
        features['ssid_popularity'] = self._calculate_ssid_popularity(df)
        
        # IP subnet analysis
        features['subnet_diversity'] = self._calculate_subnet_diversity(df)
        features['ip_range_utilization'] = self._calculate_ip_range_utilization(df)
        
        return pd.DataFrame(features)
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from the data."""
        features = {}
        
        # Basic statistics
        features['total_records'] = len(df)
        features['missing_values_ratio'] = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Only use hashable columns for duplicate detection
        hashable_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
        if hashable_cols:
            features['duplicate_records_ratio'] = (len(df) - len(df[hashable_cols].drop_duplicates())) / len(df)
        else:
            features['duplicate_records_ratio'] = 0.0
        
        # Time interval statistics
        time_intervals = df['timestamp'].diff().dropna()
        if len(time_intervals) > 0:
            features['avg_time_interval'] = time_intervals.mean().total_seconds()
            features['std_time_interval'] = time_intervals.std().total_seconds()
            features['min_time_interval'] = time_intervals.min().total_seconds()
            features['max_time_interval'] = time_intervals.max().total_seconds()
        else:
            features['avg_time_interval'] = 0
            features['std_time_interval'] = 0
            features['min_time_interval'] = 0
            features['max_time_interval'] = 0
        
        # Message length statistics
        message_lengths = df['message'].str.len()
        features['avg_message_length'] = message_lengths.mean()
        features['std_message_length'] = message_lengths.std()
        features['message_length_skewness'] = stats.skew(message_lengths)
        features['message_length_kurtosis'] = stats.kurtosis(message_lengths)
        
        # Return a DataFrame with the same index as df
        return pd.DataFrame({k: [v]*len(df) for k, v in features.items()}, index=df.index)
    
    def _extract_enhanced_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced text-based features."""
        features = {}
        
        # Basic text features
        features['word_count'] = df['message'].str.split().str.len()
        features['char_count'] = df['message'].str.len()
        features['special_char_count'] = df['message'].str.count(r'[^a-zA-Z0-9\s]')
        features['uppercase_count'] = df['message'].str.count(r'[A-Z]')
        features['lowercase_count'] = df['message'].str.count(r'[a-z]')
        features['digit_count'] = df['message'].str.count(r'\d')
        
        # Advanced text features
        features['uppercase_ratio'] = features['uppercase_count'] / (features['char_count'] + 1e-8)
        features['lowercase_ratio'] = features['lowercase_count'] / (features['char_count'] + 1e-8)
        features['digit_ratio'] = features['digit_count'] / (features['char_count'] + 1e-8)
        features['special_char_ratio'] = features['special_char_count'] / (features['char_count'] + 1e-8)
        
        # Text complexity
        features['unique_chars'] = df['message'].apply(lambda x: len(set(x)))
        features['unique_words'] = df['message'].apply(lambda x: len(set(x.split())))
        features['avg_word_length'] = df['message'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
        
        # Semantic features
        features['has_url'] = df['message'].str.contains(r'http[s]?://', regex=True).astype(int)
        features['has_email'] = df['message'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True).astype(int)
        features['has_mac_address'] = df['message'].str.contains(self.mac_pattern).astype(int)
        features['has_ip_address'] = df['message'].str.contains(self.ip_pattern).astype(int)
        
        return pd.DataFrame(features)
    
    def _extract_advanced_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced features over multiple time windows."""
        features = {}
        windows = self.feature_config['window_features'].get('windows', [5, 15, 60, 240])
        
        for window_minutes in windows:
            window_features = self._calculate_advanced_window_features(df, window_minutes)
            features.update({f'window_{window_minutes}min_{k}': v for k, v in window_features.items()})
        
        return pd.DataFrame(features)
    
    def _extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced anomaly detection features."""
        features = {}
        
        # Entropy-based features
        features['message_entropy'] = df['message'].apply(self._calculate_entropy)
        features['process_entropy'] = self._calculate_process_entropy(df)
        
        # Anomaly scores
        if self.feature_config['advanced_features'].get('include_anomaly_scores', True):
            features['message_length_anomaly'] = self._calculate_anomaly_score(df['message'].str.len())
            features['time_interval_anomaly'] = self._calculate_anomaly_score(df['timestamp'].diff().dropna().dt.total_seconds())
        
        # Correlation features
        features['mac_ip_correlation'] = self._calculate_mac_ip_correlation(df)
        features['time_event_correlation'] = self._calculate_time_event_correlation(df)
        
        return pd.DataFrame(features)
    
    # Helper methods for feature extraction
    def _extract_ssid(self, text: str) -> Optional[str]:
        """Extract SSID from text."""
        match = self.ssid_pattern.search(text)
        return match.group(1) if match else None
    
    def _extract_channel(self, text: str) -> Optional[str]:
        """Extract channel number from text."""
        match = self.channel_pattern.search(text)
        return match.group(1) if match else None
    
    def _extract_signal_strength(self, text: str) -> Optional[str]:
        """Extract signal strength from text."""
        match = self.signal_pattern.search(text)
        return match.group(1) if match else None
    
    def _calculate_message_complexity(self, message: str) -> float:
        """Calculate message complexity score."""
        if not message:
            return 0.0
        
        # Factors: length, special chars, case variation, numbers
        length_factor = min(len(message) / 100, 1.0)
        special_factor = min(message.count(' ') / len(message), 1.0) if len(message) > 0 else 0
        case_factor = abs(message.count('A-Z') - message.count('a-z')) / len(message) if len(message) > 0 else 0
        number_factor = message.count('0-9') / len(message) if len(message) > 0 else 0
        
        return (length_factor + special_factor + case_factor + number_factor) / 4
    
    def _calculate_connection_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate connection rate over time."""
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        return self._calculate_rate(connection_mask, df['timestamp'])
    
    def _calculate_disconnection_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate disconnection rate over time."""
        disconnect_mask = df['message'].str.contains('AP-STA-DISCONNECTED', case=False)
        return self._calculate_rate(disconnect_mask, df['timestamp'])
    
    def _calculate_rate(self, mask: pd.Series, timestamps: pd.Series, window_minutes: int = 60) -> pd.Series:
        """Calculate event rate over sliding window."""
        window = timedelta(minutes=window_minutes)
        rates = []
        
        for i, timestamp in enumerate(timestamps):
            window_start = timestamp - window
            window_mask = (timestamps >= window_start) & (timestamps <= timestamp)
            window_events = mask[window_mask].sum()
            window_duration = (timestamp - window_start).total_seconds() / 3600  # hours
            rate = window_events / (window_duration + 1e-8)
            rates.append(rate)
        
        return pd.Series(rates, index=timestamps.index)
    
    def _detect_burst_activity(self, df: pd.DataFrame, pattern: str, window_minutes: int = 5) -> pd.Series:
        """Detect burst activity patterns."""
        mask = df['message'].str.contains(pattern, case=False, regex=True)
        window = timedelta(minutes=window_minutes)
        burst_scores = []
        
        for i, timestamp in enumerate(df['timestamp']):
            window_start = timestamp - window
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= timestamp)
            window_events = mask[window_mask].sum()
            burst_scores.append(window_events)
        
        return pd.Series(burst_scores, index=df.index)
    
    def _calculate_unique_devices_per_window(self, df: pd.DataFrame, hours: int = 1) -> pd.Series:
        """Calculate unique devices per time window."""
        window = timedelta(hours=hours)
        device_counts = []
        
        for i, timestamp in enumerate(df['timestamp']):
            window_start = timestamp - window
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= timestamp)
            window_macs = []
            for macs in df.loc[window_mask, 'mac_addresses']:
                window_macs.extend(macs)
            device_counts.append(len(set(window_macs)))
        
        return pd.Series(device_counts, index=df.index)
    
    def _calculate_device_connection_frequency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate connection frequency per device."""
        # This is a simplified version - in practice, you'd track per device
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        return self._calculate_rate(connection_mask, df['timestamp'], window_minutes=60)
    
    def _calculate_activity_regularity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate activity regularity score."""
        # Calculate time intervals between events
        time_intervals = df['timestamp'].diff().dropna().dt.total_seconds()
        if len(time_intervals) > 1:
            regularity = 1.0 / (1.0 + time_intervals.std())
        else:
            regularity = 0.0
        
        return pd.Series([regularity] * len(df), index=df.index)
    
    def _find_peak_activity_hour(self, df: pd.DataFrame) -> pd.Series:
        """Find peak activity hour."""
        hourly_activity = df.groupby(df['timestamp'].dt.hour).size()
        peak_hour = hourly_activity.idxmax() if len(hourly_activity) > 0 else 12
        
        return pd.Series([peak_hour] * len(df), index=df.index)
    
    def _detect_unusual_connection_times(self, df: pd.DataFrame) -> pd.Series:
        """Detect connections at unusual times."""
        # Consider unusual: night hours (10 PM - 6 AM) on weekdays
        unusual_mask = (
            ((df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour < 6)) &
            (df['timestamp'].dt.dayofweek < 5)  # Weekdays
        )
        
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        return (unusual_mask & connection_mask).astype(int)
    
    def _detect_rapid_reconnections(self, df: pd.DataFrame, threshold_seconds: int = 30) -> pd.Series:
        """Detect rapid reconnections."""
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        rapid_reconnects = []
        
        for i, (timestamp, is_connection) in enumerate(zip(df['timestamp'], connection_mask)):
            if is_connection:
                # Look for previous connection within threshold
                time_diff = (timestamp - df['timestamp'].iloc[:i]).dt.total_seconds()
                recent_connections = (time_diff > 0) & (time_diff < threshold_seconds)
                rapid_reconnects.append(recent_connections.any())
            else:
                rapid_reconnects.append(False)
        
        return pd.Series(rapid_reconnects, index=df.index)
    
    # Network topology methods (simplified implementations)
    def _calculate_network_density(self, df: pd.DataFrame) -> pd.Series:
        """Calculate network density."""
        # Simplified: ratio of actual connections to possible connections
        return pd.Series([0.1] * len(df), index=df.index)  # Placeholder
    
    def _calculate_network_centralization(self, df: pd.DataFrame) -> pd.Series:
        """Calculate network centralization."""
        return pd.Series([0.5] * len(df), index=df.index)  # Placeholder
    
    def _calculate_network_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate network clustering coefficient."""
        return pd.Series([0.3] * len(df), index=df.index)  # Placeholder
    
    def _calculate_channel_utilization(self, df: pd.DataFrame) -> pd.Series:
        """Calculate channel utilization."""
        return pd.Series([0.6] * len(df), index=df.index)  # Placeholder
    
    def _calculate_channel_interference(self, df: pd.DataFrame) -> pd.Series:
        """Calculate channel interference."""
        return pd.Series([0.2] * len(df), index=df.index)  # Placeholder
    
    def _calculate_channel_switching_frequency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate channel switching frequency."""
        return pd.Series([0.1] * len(df), index=df.index)  # Placeholder
    
    def _calculate_ssid_diversity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SSID diversity."""
        return pd.Series([0.4] * len(df), index=df.index)  # Placeholder
    
    def _calculate_ssid_popularity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SSID popularity."""
        return pd.Series([0.7] * len(df), index=df.index)  # Placeholder
    
    def _calculate_subnet_diversity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate subnet diversity."""
        return pd.Series([0.3] * len(df), index=df.index)  # Placeholder
    
    def _calculate_ip_range_utilization(self, df: pd.DataFrame) -> pd.Series:
        """Calculate IP range utilization."""
        return pd.Series([0.8] * len(df), index=df.index)  # Placeholder
    
    def _calculate_advanced_window_features(self, df: pd.DataFrame, window_minutes: int) -> Dict[str, pd.Series]:
        """Calculate advanced features over a time window."""
        features = {}
        window = timedelta(minutes=window_minutes)
        
        # Event counts
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        disconnect_mask = df['message'].str.contains('AP-STA-DISCONNECTED', case=False)
        error_mask = df['message'].str.contains('ERROR|FAILED', case=False)
        
        features['connection_count'] = self._rolling_count(df, connection_mask, window)
        features['disconnect_count'] = self._rolling_count(df, disconnect_mask, window)
        features['error_count'] = self._rolling_count(df, error_mask, window)
        
        # Unique counts
        features['unique_macs'] = self._rolling_unique_count(df, 'mac_addresses', window)
        features['unique_ips'] = self._rolling_unique_count(df, 'ip_addresses', window)
        features['unique_processes'] = self._rolling_unique_count(df, 'process_name', window)
        
        # Ratios
        features['error_ratio'] = features['error_count'] / (features['connection_count'] + 1e-8)
        features['disconnect_ratio'] = features['disconnect_count'] / (features['connection_count'] + 1e-8)
        
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
            if column == 'mac_addresses' or column == 'ip_addresses':
                all_items = []
                for items in df.loc[window_mask, column]:
                    all_items.extend(items)
                counts.append(len(set(all_items)))
            else:
                counts.append(df.loc[window_mask, column].nunique())
        return pd.Series(counts, index=df.index)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate entropy of text."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_process_entropy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate process entropy."""
        process_counts = df['process_name'].value_counts()
        total_processes = len(df)
        entropy = 0.0
        
        for count in process_counts.values:
            p = count / total_processes
            if p > 0:
                entropy -= p * np.log2(p)
        
        return pd.Series([entropy] * len(df), index=df.index)
    
    def _calculate_anomaly_score(self, series: pd.Series) -> pd.Series:
        """Calculate anomaly score using z-score."""
        if len(series) < 2:
            return pd.Series([0.0] * len(series), index=series.index)
        
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val == 0:
            return pd.Series([0.0] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean_val) / std_val)
        return z_scores
    
    def _calculate_mac_ip_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate correlation between MAC and IP addresses."""
        # Simplified: return correlation based on presence
        mac_present = df['mac_addresses'].apply(len) > 0
        ip_present = df['ip_addresses'].apply(len) > 0
        correlation = (mac_present & ip_present).astype(int)
        return correlation
    
    def _calculate_time_event_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate correlation between time and events."""
        # Simplified: return hour-based correlation
        hour = df['timestamp'].dt.hour
        connection_mask = df['message'].str.contains('AP-STA-CONNECTED', case=False)
        correlation = (hour >= 9) & (hour <= 17) & connection_mask
        return correlation.astype(int)
    
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
            'is_connection_event', 'is_connect_event', 'is_disconnect_event',
            'is_auth_event', 'is_security_event', 'is_error_event', 'is_critical_error',
            'mac_count', 'ip_count', 'has_ssid', 'has_channel', 'has_signal',
            'signal_strength_numeric', 'signal_strength_bin', 'channel_numeric',
            'is_2_4ghz', 'is_5ghz', 'message_length', 'message_complexity',
            'hour_of_day', 'day_of_week', 'minute_of_hour', 'second_of_minute',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin', 'minute_cos',
            'time_since_midnight', 'time_since_week_start', 'is_weekend',
            'is_business_hours', 'is_night_hours', 'is_peak_hours',
            'connection_rate', 'disconnection_rate', 'connection_ratio',
            'connection_burst', 'error_burst', 'unique_devices_per_hour',
            'device_connection_frequency', 'activity_regularity', 'peak_activity_hour',
            'unusual_connection_time', 'rapid_reconnections'
        ] 