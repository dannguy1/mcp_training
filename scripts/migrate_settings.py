#!/usr/bin/env python3
"""
Migration script to convert existing settings to the new .env file format.
"""

import json
import os
import sys
from pathlib import Path

def migrate_settings():
    """Migrate existing settings to .env file format."""
    print("🔧 MCP Training Service Settings Migration")
    print("=" * 50)
    
    # Check for existing user_settings.json
    user_settings_path = Path("config/user_settings.json")
    if not user_settings_path.exists():
        print("❌ No existing user_settings.json found.")
        print("Creating default .env file from env.example...")
        create_default_env()
        return
    
    # Load existing settings
    try:
        with open(user_settings_path, 'r') as f:
            existing_settings = json.load(f)
        print(f"✅ Found existing settings in {user_settings_path}")
    except Exception as e:
        print(f"❌ Error reading existing settings: {e}")
        return
    
    # Create .env file from existing settings
    env_content = []
    
    # Service Configuration
    general = existing_settings.get("general", {})
    env_content.extend([
        "# Service Configuration",
        f"TRAINING_SERVICE_NAME={general.get('service_name', 'MCP Training Service')}",
        f"TRAINING_SERVICE_VERSION={general.get('service_version', '1.0.0')}",
        "TRAINING_DEBUG=false",
        ""
    ])
    
    # API Configuration
    env_content.extend([
        "# API Configuration",
        "TRAINING_API_HOST=0.0.0.0",
        "TRAINING_API_PORT=8000",
        "TRAINING_API_WORKERS=1",
        ""
    ])
    
    # Storage Configuration
    storage = existing_settings.get("storage", {})
    env_content.extend([
        "# Storage Configuration",
        f"TRAINING_MODELS_DIR={storage.get('models_dir', 'models')}",
        f"TRAINING_EXPORTS_DIR={storage.get('exports_dir', 'exports')}",
        f"TRAINING_LOGS_DIR={storage.get('logs_dir', 'logs')}",
        f"TRAINING_MAX_STORAGE_GB={storage.get('max_storage_gb', 10)}",
        f"TRAINING_AUTO_CLEANUP={str(storage.get('auto_cleanup', True)).lower()}",
        f"TRAINING_RETENTION_DAYS={storage.get('retention_days', 30)}",
        ""
    ])
    
    # Training Configuration
    training = existing_settings.get("training", {})
    env_content.extend([
        "# Training Configuration",
        "TRAINING_MAX_TRAINING_TIME=3600",
        "TRAINING_MAX_MEMORY_USAGE=4096",
        "TRAINING_ENABLE_GPU=false",
        f"TRAINING_MAX_CONCURRENT_JOBS={training.get('max_concurrent_jobs', 3)}",
        f"TRAINING_DEFAULT_MAX_ITERATIONS={training.get('default_max_iterations', 1000)}",
        f"TRAINING_DEFAULT_LEARNING_RATE={training.get('default_learning_rate', 0.01)}",
        f"TRAINING_JOB_TIMEOUT={training.get('job_timeout', 24)}",
        ""
    ])
    
    # Model Configuration
    env_content.extend([
        "# Model Configuration",
        "TRAINING_DEFAULT_MODEL_TYPE=isolation_forest",
        "TRAINING_MODEL_CONFIG_FILE=config/model_config.yaml",
        ""
    ])
    
    # Logging Configuration
    logging_config = existing_settings.get("logging", {})
    env_content.extend([
        "# Logging Configuration",
        f"TRAINING_LOG_LEVEL={logging_config.get('log_level', 'INFO')}",
        f"TRAINING_LOG_FILE={logging_config.get('log_file', 'logs/training.log')}",
        f"TRAINING_LOG_FORMAT={logging_config.get('log_format', 'structured')}",
        f"TRAINING_MAX_LOG_SIZE_MB={logging_config.get('max_log_size_mb', 100)}",
        f"TRAINING_LOG_TO_CONSOLE={str(logging_config.get('log_to_console', True)).lower()}",
        f"TRAINING_LOG_TO_FILE={str(logging_config.get('log_to_file', True)).lower()}",
        ""
    ])
    
    # General UI Configuration
    env_content.extend([
        "# General UI Configuration",
        f"TRAINING_TIMEZONE={general.get('timezone', 'UTC')}",
        f"TRAINING_DATE_FORMAT={general.get('date_format', 'YYYY-MM-DD')}",
        f"TRAINING_AUTO_REFRESH={str(general.get('auto_refresh', True)).lower()}",
        f"TRAINING_NOTIFICATIONS={str(general.get('notifications', True)).lower()}",
        ""
    ])
    
    # Security Configuration
    security = existing_settings.get("security", {})
    env_content.extend([
        "# Security Configuration",
        f"TRAINING_AUTH_ENABLED={str(security.get('auth_enabled', False)).lower()}",
        f"TRAINING_API_KEY={security.get('api_key', '')}",
        f"TRAINING_CORS_ORIGINS={security.get('cors_origins', 'http://localhost:3000,https://example.com')}",
        f"TRAINING_RATE_LIMIT={security.get('rate_limit', 100)}",
        f"TRAINING_HTTPS_ONLY={str(security.get('https_only', False)).lower()}",
        f"TRAINING_SECURE_HEADERS={str(security.get('secure_headers', True)).lower()}",
        ""
    ])
    
    # Advanced Configuration
    advanced = existing_settings.get("advanced", {})
    env_content.extend([
        "# Advanced Configuration",
        "TRAINING_ENABLE_MONITORING=true",
        "TRAINING_PROMETHEUS_PORT=9091",
        f"TRAINING_PERFORMANCE_MONITORING={str(advanced.get('performance_monitoring', True)).lower()}",
        f"TRAINING_WEBSOCKET_ENABLED={str(advanced.get('websocket_enabled', True)).lower()}",
        f"TRAINING_AUTO_BACKUP={str(advanced.get('auto_backup', False)).lower()}"
    ])
    
    # Write .env file
    env_path = Path(".env")
    try:
        with open(env_path, 'w') as f:
            f.write("\n".join(env_content))
        print(f"✅ Successfully created {env_path}")
        print(f"📝 Migrated {len(existing_settings)} setting sections")
        
        # Backup original settings
        backup_path = Path("config/user_settings.json.backup")
        with open(backup_path, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        print(f"💾 Original settings backed up to {backup_path}")
        
        print("\n🎉 Migration completed successfully!")
        print("📋 Next steps:")
        print("   1. Review the generated .env file")
        print("   2. Restart the application to use new settings")
        print("   3. Delete the backup file if everything works correctly")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return

def create_default_env():
    """Create a default .env file from env.example."""
    env_example_path = Path("env.example")
    env_path = Path(".env")
    
    if not env_example_path.exists():
        print("❌ env.example not found. Creating minimal .env file...")
        create_minimal_env()
        return
    
    try:
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {env_path} from {env_example_path}")
        print("📝 Please review and customize the settings as needed")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        create_minimal_env()

def create_minimal_env():
    """Create a minimal .env file with essential settings."""
    env_content = [
        "# MCP Training Service Configuration",
        "TRAINING_SERVICE_NAME=MCP Training Service",
        "TRAINING_API_HOST=0.0.0.0",
        "TRAINING_API_PORT=8000",
        "TRAINING_LOG_LEVEL=INFO",
        "TRAINING_DEBUG=false"
    ]
    
    try:
        with open(".env", 'w') as f:
            f.write("\n".join(env_content))
        print("✅ Created minimal .env file")
    except Exception as e:
        print(f"❌ Error creating minimal .env file: {e}")

if __name__ == "__main__":
    migrate_settings() 