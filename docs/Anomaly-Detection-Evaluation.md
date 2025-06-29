# Anomaly Detection Model Evaluation System

## Overview

The MCP Training Service evaluation system has been specifically optimized for unsupervised anomaly detection models like Isolation Forest. This system focuses on the metrics and analysis techniques that are most relevant for anomaly detection, rather than general supervised learning evaluation.

## Core Design Principles

### 1. **Unsupervised-First Approach**
- The system is designed primarily for unsupervised anomaly detection
- Ground truth labels are optional and only used for validation when available
- Core metrics focus on score distribution and model behavior

### 2. **Anomaly Detection Focus**
- Metrics are specifically chosen for anomaly detection effectiveness
- Thresholds and recommendations are tailored for anomaly detection scenarios
- Feature importance calculation is optimized for unsupervised models

### 3. **Robust Error Handling**
- Graceful degradation when ground truth is not available
- Comprehensive error handling throughout the evaluation pipeline
- Meaningful fallbacks for all calculation methods

## Evaluation Components

### 1. Basic Anomaly Metrics

**Core Metrics (Always Available):**
- `score_mean`: Average anomaly score across all samples
- `score_std`: Standard deviation of anomaly scores (higher = better separation)
- `score_min/max`: Range of anomaly scores
- `score_range`: Total range of scores (max - min)
- `score_median`: Median anomaly score
- `anomaly_ratio`: Proportion of samples flagged as anomalies
- `threshold_value`: Score threshold used for anomaly classification
- `detected_anomalies`: Number of samples classified as anomalies
- `total_samples`: Total number of samples evaluated

### 2. Advanced Anomaly Metrics

**Score Distribution Analysis:**
- Detailed percentiles (25th, 50th, 75th, 90th, 95th, 99th)
- Interquartile range (IQR) analysis
- Score skewness and variance characteristics

**Anomaly Detection Characteristics:**
- High score ratio (proportion of samples above 90th percentile)
- Low score ratio (proportion of samples below 10th percentile)
- Score separation quality

**Ground Truth Validation (When Available):**
- Normal vs anomaly score statistics
- Score separation quality metrics
- Classification performance metrics (precision, recall, F1, ROC AUC)

### 3. Threshold Analysis

**Multiple Threshold Levels:**
- Analyzes performance at different percentile thresholds (50th, 75th, 80th, 85th, 90th, 95th, 99th)
- For each threshold, calculates:
  - Anomaly ratio
  - Number of detected anomalies
  - Classification metrics (if ground truth available)

### 4. Cross-Validation

**Unsupervised Cross-Validation:**
- Uses `neg_mean_squared_error` scoring
- Focuses on model consistency across folds
- Calculates cross-validation consistency metric

### 5. Feature Importance

**Unsupervised Feature Importance:**
- Calculates importance based on score variance changes when features are shuffled
- Normalized importance scores
- Fallback to uniform importance if calculation fails

## Quality Thresholds

### Anomaly Detection Thresholds

1. **Score Standard Deviation** (`score_std` ≥ 0.01)
   - Ensures meaningful score separation
   - Lower values indicate poor discrimination

2. **Anomaly Ratio** (`anomaly_ratio` ≤ 0.15)
   - Keeps anomaly detection reasonable
   - Higher ratios suggest overly sensitive model

3. **Score Range** (`score_range` ≥ 0.1)
   - Ensures sufficient score spread
   - Lower ranges indicate limited separation capability

4. **Score Distribution** (`score_std` > 0.01)
   - Validates score distribution quality
   - Essential for meaningful anomaly detection

5. **Anomaly Detection Capability** (`score_range` > 0.1)
   - Overall capability assessment
   - Combines multiple factors for comprehensive evaluation

## Performance Analysis

### Core Analysis Components

1. **Score Analysis:**
   - Variance, skewness, range, median, IQR
   - Focus on distribution characteristics

2. **Anomaly Detection Analysis:**
   - Score distribution statistics
   - Anomaly score characteristics
   - High/low score ratios

3. **Ground Truth Validation (Optional):**
   - Normal vs anomaly score comparisons
   - Separation quality metrics
   - Validation of model effectiveness

## Recommendations System

### Anomaly Detection Specific Recommendations

1. **Score Distribution Issues:**
   - Low variance → Feature engineering or parameter tuning
   - Small range → Adjust contamination or features

2. **Anomaly Ratio Issues:**
   - Too high → Decrease contamination parameter
   - Too low → Increase contamination parameter

3. **Model Performance:**
   - Below thresholds → Retrain with different parameters
   - Low consistency → More data or simpler model

4. **Feature Importance:**
   - Low importance features → Consider feature selection
   - Poor separation → Feature engineering

5. **Ground Truth Validation:**
   - Poor separation → Model tuning
   - Over-separation → Regularization

## Usage Examples

### Basic Evaluation (No Ground Truth)
```python
evaluator = ModelEvaluator(config)
results = evaluator.evaluate_model(model, X)
```

### Evaluation with Ground Truth
```python
evaluator = ModelEvaluator(config)
results = evaluator.evaluate_model(model, X, y)
```

### Accessing Results
```python
# Basic metrics
basic_metrics = results['basic_metrics']
anomaly_ratio = basic_metrics['anomaly_ratio']
score_std = basic_metrics['score_std']

# Advanced metrics
advanced_metrics = results['advanced_metrics']
score_distribution = advanced_metrics['score_distribution']
threshold_analysis = advanced_metrics['threshold_analysis']

# Performance analysis
performance = results['performance_analysis']
score_analysis = performance['score_analysis']

# Threshold checks
threshold_checks = results['threshold_checks']
passed_checks = sum(threshold_checks.values())
```

## Benefits of This Approach

### 1. **Correctness for Anomaly Detection**
- Metrics are specifically chosen for anomaly detection effectiveness
- No confusion between supervised and unsupervised evaluation
- Focus on what matters for anomaly detection

### 2. **Robustness**
- Works with or without ground truth
- Graceful error handling throughout
- Meaningful fallbacks for all calculations

### 3. **Actionable Insights**
- Specific recommendations for anomaly detection models
- Clear quality thresholds
- Practical guidance for model improvement

### 4. **Performance**
- Optimized calculations for anomaly detection
- Efficient feature importance calculation
- Minimal computational overhead

## Integration with Training Pipeline

The evaluation system integrates seamlessly with the training pipeline:

1. **Automatic Evaluation**: Runs after model training
2. **Quality Assessment**: Checks if model meets requirements
3. **Recommendations**: Provides actionable feedback
4. **Metadata Storage**: Saves evaluation results with model

This ensures that every trained model is properly evaluated and validated for anomaly detection effectiveness. 