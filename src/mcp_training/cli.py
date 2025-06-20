"""
Command-line interface for MCP Training Service.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional
import logging

from .core.config import get_global_config
from .core.feature_extractor import WiFiFeatureExtractor
from .core.model_trainer import ModelTrainer
from .core.export_validator import ExportValidator

# Set up logging
logging.basicConfig(
    level=getattr(logging, get_global_config().log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=get_global_config().service_version)
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug):
    """MCP Training Service CLI"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        get_global_config().debug = True


@cli.command()
@click.argument('export_file', type=click.Path(exists=True))
@click.option('--model-type', default=get_global_config().default_model_type, 
              help='Type of model to train')
@click.option('--output-name', help='Name for the trained model')
@click.option('--validate-only', is_flag=True, help='Only validate the export file')
def train(export_file: str, model_type: str, output_name: Optional[str], validate_only: bool):
    """Train a model from export data."""
    try:
        # Validate export file
        validator = ExportValidator()
        is_valid, errors = validator.validate_export_file(export_file)
        
        if not is_valid:
            click.echo("❌ Export file validation failed:")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)
        
        click.echo("✅ Export file validation passed")
        
        if validate_only:
            # Show export summary
            summary = validator.get_export_summary(export_file)
            if summary:
                click.echo("\n📊 Export Summary:")
                click.echo(f"  Records: {summary['record_count']}")
                click.echo(f"  File size: {summary['file_size_mb']:.2f} MB")
                click.echo(f"  Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
                click.echo(f"  Duration: {summary['time_range']['duration_hours']:.1f} hours")
            return
        
        # Load export data
        click.echo("📂 Loading export data...")
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        data = export_data.get('data', [])
        if not data:
            click.echo("❌ No data found in export file")
            sys.exit(1)
        
        click.echo(f"📊 Found {len(data)} records")
        
        # Train model
        click.echo(f"🤖 Training {model_type} model...")
        trainer = ModelTrainer(model_type=model_type)
        
        model_name = output_name or f"{model_type}_{Path(export_file).stem}"
        results = trainer.train(data, model_name)
        
        # Display results
        click.echo("✅ Training completed successfully!")
        click.echo(f"\n📋 Training Results:")
        click.echo(f"  Model name: {results['model_name']}")
        click.echo(f"  Model type: {results['model_type']}")
        click.echo(f"  Data points: {results['data_points']}")
        click.echo(f"  Features: {results['features_count']}")
        click.echo(f"  Model path: {results['model_path']}")
        
        # Evaluation results
        eval_results = results['evaluation']
        click.echo(f"\n📈 Evaluation Results:")
        score_stats = eval_results['score_statistics']
        click.echo(f"  Mean score: {score_stats['mean_score']:.4f}")
        click.echo(f"  Score std: {score_stats['std_score']:.4f}")
        click.echo(f"  Score range: {score_stats['min_score']:.4f} to {score_stats['max_score']:.4f}")
        
        if eval_results['cross_validation_score']:
            click.echo(f"  CV score: {eval_results['cross_validation_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"❌ Training failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('export_file', type=click.Path(exists=True))
def validate(export_file: str):
    """Validate an export file."""
    try:
        validator = ExportValidator()
        is_valid, errors = validator.validate_export_file(export_file)
        
        if is_valid:
            click.echo("✅ Export file is valid")
            
            # Show summary
            summary = validator.get_export_summary(export_file)
            if summary:
                click.echo(f"\n📊 File Summary:")
                click.echo(f"  Records: {summary['record_count']}")
                click.echo(f"  File size: {summary['file_size_mb']:.2f} MB")
                if summary['time_range']:
                    click.echo(f"  Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
                    click.echo(f"  Duration: {summary['time_range']['duration_hours']:.1f} hours")
                
                click.echo(f"\n📈 Data Distribution:")
                click.echo(f"  Log levels: {dict(summary['log_level_distribution'])}")
                click.echo(f"  Top processes: {dict(list(summary['process_distribution'].items())[:5])}")
        else:
            click.echo("❌ Export file validation failed:")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"❌ Validation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--export-dir', default=get_global_config().exports_dir, 
              help='Directory containing export files')
def validate_all(export_dir: str):
    """Validate all export files in a directory."""
    try:
        validator = ExportValidator()
        results = validator.validate_multiple_exports(export_dir)
        
        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            sys.exit(1)
        
        click.echo(f"📁 Directory: {results['directory']}")
        click.echo(f"📊 Total files: {results['total_files']}")
        click.echo(f"✅ Valid files: {results['valid_files']}")
        click.echo(f"❌ Invalid files: {results['invalid_files']}")
        
        if results['invalid_files'] > 0:
            click.echo(f"\n❌ Invalid files:")
            for file_result in results['files']:
                if not file_result['is_valid']:
                    click.echo(f"  - {file_result['file_name']}:")
                    for error in file_result['errors'][:3]:  # Show first 3 errors
                        click.echo(f"    {error}")
                    if len(file_result['errors']) > 3:
                        click.echo(f"    ... and {len(file_result['errors']) - 3} more errors")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"❌ Validation failed: {e}")
        sys.exit(1)


@cli.command()
def list_models():
    """List all trained models."""
    try:
        trainer = ModelTrainer()
        models = trainer.list_models()
        
        if not models:
            click.echo("📭 No trained models found")
            return
        
        click.echo(f"🤖 Found {len(models)} trained models:")
        click.echo()
        
        for model in models:
            click.echo(f"📋 {model['name']}")
            click.echo(f"   Type: {model['model_type']}")
            click.echo(f"   Training date: {model['training_date']}")
            click.echo(f"   Features: {model['feature_count']}")
            click.echo()
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        click.echo(f"❌ Failed to list models: {e}")
        sys.exit(1)


@cli.command()
@click.argument('model_name')
def delete_model(model_name: str):
    """Delete a trained model."""
    try:
        trainer = ModelTrainer()
        
        if click.confirm(f"Are you sure you want to delete model '{model_name}'?"):
            success = trainer.delete_model(model_name)
            if success:
                click.echo(f"✅ Model '{model_name}' deleted successfully")
            else:
                click.echo(f"❌ Failed to delete model '{model_name}'")
                sys.exit(1)
        else:
            click.echo("❌ Deletion cancelled")
            
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        click.echo(f"❌ Failed to delete model: {e}")
        sys.exit(1)


@cli.command()
@click.argument('export_file', type=click.Path(exists=True))
@click.argument('model_name')
def predict(export_file: str, model_name: str):
    """Make predictions using a trained model."""
    try:
        # Load model
        trainer = ModelTrainer()
        model_path = Path(get_global_config().models_dir) / model_name
        
        if not model_path.exists():
            click.echo(f"❌ Model '{model_name}' not found")
            sys.exit(1)
        
        if not trainer.load_model(str(model_path)):
            click.echo(f"❌ Failed to load model '{model_name}'")
            sys.exit(1)
        
        click.echo(f"✅ Model '{model_name}' loaded successfully")
        
        # Load export data
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        data = export_data.get('data', [])
        if not data:
            click.echo("❌ No data found in export file")
            sys.exit(1)
        
        click.echo(f"📊 Making predictions on {len(data)} records...")
        
        # Get predictions
        predictions = trainer.predict(data)
        scores = trainer.get_anomaly_scores(data)
        
        # Analyze results
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        click.echo(f"\n📈 Prediction Results:")
        click.echo(f"  Normal records: {normal_count}")
        click.echo(f"  Anomaly records: {anomaly_count}")
        click.echo(f"  Anomaly rate: {anomaly_count / len(data) * 100:.2f}%")
        click.echo(f"  Average score: {scores.mean():.4f}")
        click.echo(f"  Score std: {scores.std():.4f}")
        
        # Show top anomalies
        if anomaly_count > 0:
            anomaly_indices = (predictions == -1).nonzero()[0]
            click.echo(f"\n🚨 Top anomalies (first 5):")
            for i, idx in enumerate(anomaly_indices[:5]):
                record = data[idx]
                click.echo(f"  {i+1}. Score: {scores[idx]:.4f}")
                click.echo(f"     Time: {record.get('timestamp', 'N/A')}")
                click.echo(f"     Process: {record.get('process_name', 'N/A')}")
                click.echo(f"     Message: {record.get('message', 'N/A')[:100]}...")
                click.echo()
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"❌ Prediction failed: {e}")
        sys.exit(1)


@cli.command()
def info():
    """Show system information."""
    try:
        click.echo(f"🤖 MCP Training Service v{get_global_config().service_version}")
        click.echo(f"📁 Models directory: {get_global_config().models_dir}")
        click.echo(f"📁 Exports directory: {get_global_config().exports_dir}")
        click.echo(f"📁 Logs directory: {get_global_config().logs_dir}")
        click.echo(f"🔧 Default model type: {get_global_config().default_model_type}")
        click.echo(f"⚙️  Debug mode: {get_global_config().debug}")
        
        # Check directories
        models_path = Path(get_global_config().models_dir)
        exports_path = Path(get_global_config().exports_dir)
        logs_path = Path(get_global_config().logs_dir)
        
        click.echo(f"\n📂 Directory Status:")
        click.echo(f"  Models: {'✅' if models_path.exists() else '❌'} {models_path}")
        click.echo(f"  Exports: {'✅' if exports_path.exists() else '❌'} {exports_path}")
        click.echo(f"  Logs: {'✅' if logs_path.exists() else '❌'} {logs_path}")
        
        # Count models
        if models_path.exists():
            model_count = len([d for d in models_path.iterdir() if d.is_dir()])
            click.echo(f"\n🤖 Trained models: {model_count}")
        
        # Count exports
        if exports_path.exists():
            export_count = len(list(exports_path.glob('*.json')))
            click.echo(f"📄 Export files: {export_count}")
        
    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        click.echo(f"❌ Failed to get info: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli() 