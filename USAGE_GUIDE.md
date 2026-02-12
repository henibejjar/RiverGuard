# RiverGuard Usage Guide

🌐 **[Try Live Demo](https://heni-bejjar-riverguard.streamlit.app/)**

## Quick Start (Using Pre-trained Models)

**All trained models are included** in this repository. No dataset download or training required!

```bash
pip install -r requirements.txt
streamlit run app_flood.py
```

Open `http://localhost:8501` and start predicting immediately.

## Full Training (Optional)

Only needed if you want to retrain models with custom hyperparameters or verify reproducibility.

### 1. Download Dataset

Download from [Kaggle - Flood Prediction Train and Test Dataset](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset):
- `train.csv` (20 base features + `FloodProbability`) — ~1.4M rows
- `test.csv` (20 base features + `id`) — For batch predictions

Place both files in the project root directory.

### 2. Train Models

```bash
pip install -r requirements.txt
python flood_prediction_model.py  # Takes 5-10 minutes
streamlit run app_flood.py
```

## App Modes

1) **Quick Prediction**
- Enter 20 features (sliders or text)
- Input summary + stats
- Prediction gauge + risk label
- **95% Confidence Interval** (uncertainty bounds)
- **Top 5 Feature Impact** (most influential factors)
- Model comparison across all 4 approaches

2) **Advanced Analysis**
- Upload a dataset
- Dataset-based sensitivity, distribution, correlation

3) **Batch Processing**
- Upload CSV (up to 200MB, 1.4M+ rows tested)
- **Values are auto-scaled** using the saved scaler (raw data accepted)
- Missing values auto-imputed with medians
- Metrics if target column exists
- Download predictions

4) **Help & Info** (5 tabs)
- Tab 1: Normalization explanation
- Tab 2: Detailed feature dictionary (searchable)
- Tab 3: Model approach explanations
- Tab 4: Troubleshooting guide
- Tab 5: Training dataset overview (correlation, sensitivity, distribution)

## Included Model Files

This repository includes all pre-trained models (~14 MB):
- `approach1_models.pkl` - Blending ensemble (4.05 MB)
- `approach2_resnet.pt` - ResNet neural network (2.13 MB)
- `approach3_models.pkl` - Stacking ensemble (2.58 MB)
- `flood_scaler.pkl` - Feature scaler (required for predictions)
- `flood_model_metadata.pkl` - Training metadata
- `all_predictions.pkl` - Predictions archive (4.95 MB)
- `train_correlation_matrix.pkl` - Pre-computed correlation matrix
- `train_sensitivity_approach4.pkl` - Pre-computed sensitivity scores

**Ready to use immediately!** Dataset files only needed for retraining.

## Common Issues

- **Upload errors in Batch Processing:** Ensure CSV has all 20 feature columns
- **Missing target column:** Metrics will be skipped (predictions still work)
- **Want to retrain models:** Download datasets from Kaggle, run `python flood_prediction_model.py`

---

## Data Format Details

**Target column:**
- `FloodProbability` (values expected in the range [0, 1])

**Optional column:**
- `id` (ignored during training if present)

**Data expectations:**
- Numeric values for all 20 features
- Missing values are allowed (the training pipeline fills them with medians)
- Feature values are expected to be in [0, 1] for the app UI, but training can accept broader numeric ranges

---

## Approach Explanations (Techniques and Rationale)

### Approach 1: Blending + Ridge Meta-Learner
**Techniques used:**
- LightGBM, CatBoost, XGBoost, HistGradientBoosting
- Ridge regression as a meta-learner

**Why this approach:**
- Multiple gradient-boosted trees capture different nonlinear patterns
- Blending reduces variance and stabilizes predictions
- Ridge meta-learner combines model outputs with regularization to avoid overfitting

---

### Approach 2: ResNet Tabular Neural Network
**Techniques used:**
- Residual blocks (skip connections)
- Batch normalization + dropout
- Adam optimizer with MSE loss

**Why this approach:**
- Residual connections improve training stability in deep MLPs
- Dropout + batch norm improves generalization
- Good at learning complex feature interactions not easily captured by trees

---

### Approach 3: Stacking (ResNet + CatBoost + XGBoost) + Ridge
**Techniques used:**
- Hybrid stacking with neural + tree models
- Ridge regression as a meta-learner

**Why this approach:**
- Combines complementary strengths: deep learning interactions + strong tree baselines
- Stacking typically improves accuracy over any single model
- Ridge meta-learner keeps the blend stable and interpretable

---

### Approach 4: Final Ensemble (Weighted Voting)
**Techniques used:**
- Weighted average of Approach 3 and Approach 2
- 70% weight to Approach 3, 30% weight to Approach 2

**Why this approach:**
- Approach 3 (stacking) provides strong baseline accuracy
- Approach 2 (ResNet) adds neural network diversity
- Weighted voting boosts robustness without increasing complexity
- **Recommended default** for all modes; provides best overall balance of accuracy and stability

---

## Which Model Should I Use?

- **Approach 4 (Final Ensemble):** best overall accuracy and stability (recommended default)
- **Approach 1 (Blending):** fast predictions, good baseline
- **Approach 2 (ResNet):** captures complex feature interactions, neural network approach
- **Approach 3 (Stacking):** strong hybrid combining neural and tree models

**For most use cases:** Approach 4 (used automatically in Quick Prediction and Batch Processing)

**For research/comparison:** Use Advanced Analysis to evaluate all 4 approaches side-by-side

---

## Performance Expectations

### Batch Processing Metrics

When you upload a CSV with a `FloodProbability` column, the app computes:

- **Accuracy**: % of correct risk level classifications (Low/Med/High)
- **RMSE** (Root Mean Squared Error): Typical error magnitude (0-1 scale)
- **MAE** (Mean Absolute Error): Average absolute difference from true probability
- **R²** (R-Squared): Coefficient of determination

Example interpretation:
- R² = 0.85: Model explains 85% of variance in flood probability
- RMSE = 0.12: Average prediction error is ±0.12 on probability scale
- MAE = 0.09: On average, predictions differ by 0.09 from true values

### Typical Accuracy Range
- **Accuracy (risk classification)**: 75-90%
- **RMSE (probability)**: 0.08-0.15
- **R² Score**: 0.75-0.90

These metrics depend on your data quality and how representative it is of actual flood scenarios.

---

## Real-Time Analytics Features

### Confidence Intervals (Quick Prediction)
- Displays 95% confidence bounds on each prediction
- Calculated from predictions of all 4 model approaches
- Format: `[Lower%] - [Upper%] ± Margin`
- Helps assess prediction reliability and uncertainty

### Feature Impact Visualization
- Shows top 5 most influential features for predictions
- Impact scores pre-computed on 1.4M training rows
- Based on perturbation-based sensitivity analysis (±0.1 feature change)
- Updated with every prediction for context-specific insights

### Pre-computed Analytics (Help & Info → Tab 5)
- **Correlation Matrix:** Instant feature-to-feature correlations from training data
- **Feature Sensitivity:** Pre-computed impact scores for all 20 base features  
- **Target Distribution:** FloodProbability histogram with KDE curve from training data
- **Feature-Target Correlation:** Pearson correlations between each feature and flood probability
- Enables instant loading and analysis without computation delays

## Advanced Tips

1. **For sensitive decisions:** Use Advanced Analysis mode to see all 4 approach outputs before deciding
2. **For bulk scoring:** Batch Processing with metrics helps validate model performance on your data
3. **For understanding predictions:** Check "Top Features Impact" section to see which factors drove each prediction
4. **For confidence assessment:** Use the 95% Confidence Interval in Quick Prediction to gauge uncertainty
5. **For data exploration:** Use Help tab → Dataset Overview to see correlation and sensitivity pre-computed from 1.4M rows
6. **For exploring relationships:** Use Advanced Analysis Sensitivity tab to understand which features most affect predictions across your dataset
7. **For retraining:** Modify `flood_prediction_model.py` hyperparameters and rerun the training script
8. **For pre-computation:** Run:
   ```bash
   python precompute_correlation.py
   python precompute_sensitivity.py
   ```
   before app deployment for instant analytics loading
9. **For feature engineering:** App automatically transforms 20 base features into 67 engineered features (statistical, quantile, dispersion, norm, sorted, custom)

---

## Feature Engineering Details

The app automatically transforms your 20 base input features into 67 engineered features that capture:

- **Row statistics**: Sum, mean, std, skew, kurtosis, max, min, range
- **Quantile features**: 1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th percentiles
- **Dispersion metrics**: IQR, coefficient of variation, MAD (median absolute deviation)
- **Norms & entropy**: L1 norm, L2 norm, entropy
- **Sorted variants**: Features sorted in ascending and descending order
- **Custom coefficients**: Domain-specific feature combinations (CCP, TTF)

This automatic enhancement is transparent to you—just input your 20 features, and the models use the engineered features internally.

---
