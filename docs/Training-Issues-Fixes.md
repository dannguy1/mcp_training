# Training Issues Fixes

## Overview
This document summarizes the issues identified in the training system logs and the fixes implemented to resolve them.

## Issues Identified

### 1. Permutation Importance Calculation Error
**Problem**: The evaluation system was trying to calculate permutation importance for unsupervised models using scikit-learn's `permutation_importance` function, which requires target values (`y` parameter) that don't exist in unsupervised learning.

**Error Message**:
```
"Permutation importance calculation failed: missing a required argument: 'y'"
```

**Root Cause**: The evaluation module was using supervised learning metrics for unsupervised models.

### 2. Missing Deployment Metadata
**Problem**: The model registry was repeatedly trying to load deployment metadata that didn't exist, causing unnecessary warnings.

**Error Message**:
```
"Failed to load metadata for models/deployments: Metadata file not found: models/deployments/metadata.json"
```

**Root Cause**: The deployment metadata file was missing from the expected location.

## Fixes Implemented

### 1. Fixed Unsupervised Feature Importance Calculation

**File**: `src/mcp_training/models/evaluation.py`

**Changes**:
- Replaced the problematic `permutation_importance` function with a custom unsupervised feature importance calculation
- The new approach calculates feature importance based on how much the model's predictions change when each feature is shuffled
- Added proper error handling and fallback to uniform importance scores
- Normalized importance scores for consistency

**New Implementation**:
```python
def _calculate_permutation_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
    """Calculate permutation importance for unsupervised models."""
    try:
        # For unsupervised models, we can't use standard permutation importance
        # Instead, we'll calculate feature importance based on how much the model's
        # predictions change when we shuffle each feature
        
        # Use a small sample for efficiency
        sample_size = min(1000, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_indices]
        
        # Get baseline scores
        baseline_scores = model.score_samples(X_sample)
        
        # Calculate importance for each feature
        feature_importance = {}
        for feature_idx in range(X_sample.shape[1]):
            # Create a copy of the data with this feature shuffled
            X_shuffled = X_sample.copy()
            np.random.shuffle(X_shuffled[:, feature_idx])
            
            # Get scores with shuffled feature
            shuffled_scores = model.score_samples(X_shuffled)
            
            # Calculate importance as the difference in score variance
            # Higher difference means the feature is more important
            baseline_var = np.var(baseline_scores)
            shuffled_var = np.var(shuffled_scores)
            importance = abs(baseline_var - shuffled_var)
            
            feature_importance[feature_idx] = float(importance)
        
        # Normalize importance scores
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        return feature_importance
        
    except Exception as e:
        logger.warning(f"Unsupervised feature importance calculation failed: {e}")
        # Return uniform importance as fallback
        return {i: 1.0 / X.shape[1] for i in range(X.shape[1])}
```

### 2. Created Missing Deployment Metadata

**File**: `models/deployments/metadata.json`

**Changes**:
- Created the missing deployment metadata file with proper structure
- Added initialization data to prevent registry warnings

**Content**:
```json
{
    "deployments": [],
    "last_updated": "2025-06-29T02:50:00.000000",
    "version": "1.0.0",
    "description": "Deployment metadata for MCP Training Service models"
}
```

### 3. Enhanced Error Handling and Unsupervised Model Support

**Additional Improvements**:

1. **Cross-validation**: Enhanced to handle unsupervised models properly
2. **Advanced metrics**: Added checks for ground truth availability before calculating classification metrics
3. **Threshold analysis**: Improved to work with unsupervised models
4. **Performance analysis**: Added separate analysis paths for supervised vs unsupervised models

**Key Changes**:
- Added `has_ground_truth` checks throughout the evaluation pipeline
- Implemented graceful fallbacks when ground truth is not available
- Enhanced logging to provide better feedback about evaluation mode
- Improved error handling to prevent evaluation failures

## Testing

The fixes have been tested to ensure:
- Evaluation module imports successfully
- No more permutation importance errors
- No more missing metadata warnings
- Proper handling of both supervised and unsupervised models

## Impact

These fixes resolve the training issues by:
1. **Eliminating the permutation importance error** that was causing evaluation failures
2. **Removing the metadata warnings** that were cluttering the logs
3. **Improving robustness** of the evaluation pipeline for unsupervised models
4. **Maintaining backward compatibility** with supervised models

The training system should now work smoothly without the previously reported issues. 