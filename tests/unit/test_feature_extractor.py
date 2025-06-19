"""
Unit tests for WiFiFeatureExtractor.
"""

import pytest
import pandas as pd
import numpy as np
from mcp_training.core.feature_extractor import WiFiFeatureExtractor


class TestWiFiFeatureExtractor:
    """Test cases for WiFiFeatureExtractor."""
    
    def test_init(self):
        """Test feature extractor initialization."""
        extractor = WiFiFeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'mac_pattern')
        assert hasattr(extractor, 'ip_pattern')
    
    def test_extract_mac_addresses(self):
        """Test MAC address extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Test valid MAC addresses
        text = "AP-STA-CONNECTED 00:11:22:33:44:55 and AA:BB:CC:DD:EE:FF"
        macs = extractor._extract_mac_addresses(text)
        assert len(macs) == 2
        assert "00:11:22:33:44:55" in macs
        assert "AA:BB:CC:DD:EE:FF" in macs
        
        # Test no MAC addresses
        text = "No MAC addresses here"
        macs = extractor._extract_mac_addresses(text)
        assert len(macs) == 0
    
    def test_extract_ip_addresses(self):
        """Test IP address extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Test valid IP addresses
        text = "Connection from 192.168.1.100 to 10.0.0.1"
        ips = extractor._extract_ip_addresses(text)
        assert len(ips) == 2
        assert "192.168.1.100" in ips
        assert "10.0.0.1" in ips
        
        # Test no IP addresses
        text = "No IP addresses here"
        ips = extractor._extract_ip_addresses(text)
        assert len(ips) == 0
    
    def test_extract_time_features(self):
        """Test time feature extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Create sample data
        data = [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "message": "test message",
                "process_name": "test",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T18:30:00Z",
                "message": "test message 2",
                "process_name": "test",
                "log_level": "INFO"
            }
        ]
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = extractor._extract_time_features(df)
        
        assert 'hour_of_day' in features.columns
        assert 'day_of_week' in features.columns
        assert 'minute_of_hour' in features.columns
        assert 'time_since_midnight' in features.columns
        assert 'is_weekend' in features.columns
        assert 'is_business_hours' in features.columns
        
        # Check specific values
        assert features.iloc[0]['hour_of_day'] == 12
        assert features.iloc[1]['hour_of_day'] == 18
        assert features.iloc[0]['is_business_hours'] == 1  # 12 PM is business hours
        assert features.iloc[1]['is_business_hours'] == 0  # 6:30 PM is not business hours
    
    def test_extract_wifi_features(self):
        """Test WiFi feature extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Create sample data
        data = [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "message": "AP-STA-CONNECTED 00:11:22:33:44:55",
                "process_name": "hostapd",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T12:01:00Z",
                "message": "ERROR: Authentication failed",
                "process_name": "hostapd",
                "log_level": "ERROR"
            }
        ]
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = extractor._extract_wifi_features(df)
        
        assert 'is_connection_event' in features.columns
        assert 'is_auth_event' in features.columns
        assert 'is_error_event' in features.columns
        assert 'mac_count' in features.columns
        assert 'ip_count' in features.columns
        assert 'message_length' in features.columns
        
        # Check specific values
        assert features.iloc[0]['is_connection_event'] == 1
        assert features.iloc[0]['is_error_event'] == 0
        assert features.iloc[1]['is_error_event'] == 1
        assert features.iloc[0]['mac_count'] == 1
        assert features.iloc[1]['mac_count'] == 0
    
    def test_extract_text_features(self):
        """Test text feature extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Create sample data
        data = [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "message": "Simple message",
                "process_name": "test",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T12:01:00Z",
                "message": "COMPLEX MESSAGE with numbers 123 and symbols @#$",
                "process_name": "test",
                "log_level": "INFO"
            }
        ]
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = extractor._extract_text_features(df)
        
        assert 'word_count' in features.columns
        assert 'special_char_count' in features.columns
        assert 'uppercase_ratio' in features.columns
        assert 'number_count' in features.columns
        assert 'unique_chars' in features.columns
        
        # Check specific values
        assert features.iloc[0]['word_count'] == 2
        assert features.iloc[1]['word_count'] == 7
        assert features.iloc[0]['special_char_count'] == 0
        assert features.iloc[1]['special_char_count'] > 0
        assert features.iloc[1]['number_count'] == 3
    
    def test_extract_process_features(self):
        """Test process feature extraction."""
        extractor = WiFiFeatureExtractor()
        
        # Create sample data
        data = [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "message": "message 1",
                "process_name": "hostapd",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T12:01:00Z",
                "message": "message 2",
                "process_name": "hostapd",
                "log_level": "ERROR"
            },
            {
                "timestamp": "2024-01-01T12:02:00Z",
                "message": "message 3",
                "process_name": "wpa_supplicant",
                "log_level": "WARNING"
            }
        ]
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = extractor._extract_process_features(df)
        
        assert 'process_rank' in features.columns
        assert 'log_level_numeric' in features.columns
        assert 'process_frequency' in features.columns
        
        # Check log level encoding
        assert features.iloc[0]['log_level_numeric'] == 1  # INFO
        assert features.iloc[1]['log_level_numeric'] == 3  # ERROR
        assert features.iloc[2]['log_level_numeric'] == 2  # WARNING
    
    def test_extract_features_integration(self, sample_export_data):
        """Test full feature extraction integration."""
        extractor = WiFiFeatureExtractor()
        
        data = sample_export_data['data']
        features = extractor.extract_features(data)
        
        # Check that features were extracted
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        assert len(features.columns) > 0
        
        # Check for expected feature columns
        expected_features = [
            'hour_of_day', 'day_of_week', 'is_connection_event',
            'is_error_event', 'message_length', 'word_count'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_feature_names(self):
        """Test feature names list."""
        extractor = WiFiFeatureExtractor()
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Check for expected feature names
        expected_features = [
            'hour_of_day', 'day_of_week', 'is_connection_event',
            'is_error_event', 'message_length', 'word_count'
        ]
        
        for feature in expected_features:
            assert feature in feature_names 