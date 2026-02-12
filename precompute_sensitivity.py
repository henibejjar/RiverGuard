"""
Pre-compute feature sensitivity on ALL train.csv rows (1.4M+) using Approach 4
Run this once: python precompute_sensitivity.py
"""

import pandas as pd
import numpy as np
import joblib
import sys

print("Loading models...")
scaler = joblib.load('flood_scaler.pkl')
metadata = joblib.load('flood_model_metadata.pkl')
approach1_models = joblib.load('approach1_models.pkl')
approach3_models = joblib.load('approach3_models.pkl')

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)

class ResNetTabular(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        for block in self.blocks:
            out = block(out)
        out = torch.sigmoid(self.output_layer(out))
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resnet_model = ResNetTabular(input_dim=metadata['n_features'], hidden_dim=256, num_blocks=4)
resnet_state = torch.load('approach2_resnet.pt', map_location=device, weights_only=False)
resnet_model.load_state_dict(resnet_state)
resnet_model.to(device)
resnet_model.eval()

models_dict = {
    'scaler': scaler,
    'metadata': metadata,
    'approach1': approach1_models,
    'approach2': resnet_model,
    'approach3': approach3_models,
    'device': device
}

print("Loading train.csv...")
base_features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
    'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
]

train_df = pd.read_csv('train.csv', usecols=base_features)
print(f"Loaded {len(train_df):,} rows")

def advanced_features(df):
    """Create advanced engineered features from base features"""
    from scipy.stats import skew, kurtosis, entropy
    df = df.copy()

    # CRITICAL: Fill NaN values BEFORE feature engineering
    for col in base_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0.5)
    
    data = df[base_features].values

    # Row-wise Statistics
    df['sum'] = np.nansum(data, axis=1)
    df['mean'] = np.nanmean(data, axis=1)
    df['std'] = np.nanstd(data, axis=1)
    df['std'] = df['std'].fillna(0)
    df['skew'] = skew(data, axis=1, nan_policy='omit')
    df['kurt'] = kurtosis(data, axis=1, nan_policy='omit')
    df['max'] = np.nanmax(data, axis=1)
    df['min'] = np.nanmin(data, axis=1)
    df['range'] = df['max'] - df['min']
    df['sum_sq'] = np.nansum(data**2, axis=1)
    df['ptp'] = np.ptp(data, axis=1)

    # Quantiles
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        df[f'q{int(q*100):02d}'] = np.quantile(data, q, axis=1, method='linear')

    # Dispersion & Robustness
    df['iqr'] = df['q75'] - df['q25']
    df['cv'] = df['std'] / (df['mean'] + 1e-6)
    df['mad'] = np.mean(np.abs(data - df['mean'].values[:, None]), axis=1)

    # Norm-based & Distribution
    df['l1_norm'] = np.sum(np.abs(data), axis=1)
    df['l2_norm'] = np.linalg.norm(data, axis=1)
    
    # Entropy with nan_policy
    entropy_vals = []
    for row in data:
        row_clean = row[~np.isnan(row)]
        if len(row_clean) > 0:
            entropy_vals.append(entropy(np.abs(row_clean) + 1e-9))
        else:
            entropy_vals.append(0)
    df['entropy'] = entropy_vals

    # Sorted Features
    sorted_features = np.sort(data, axis=1)
    for i in range(len(base_features)):
        df[f'sort_{i}'] = sorted_features[:, i]

    # Custom Coefficients
    df['CCP'] = (df['mean'] / (df['std'] + 1e-6)) * (df['q50'] / (df['max'] + 1e-6))
    df['top3_mean'] = sorted_features[:, -3:].mean(axis=1)
    df['bottom3_mean'] = sorted_features[:, :3].mean(axis=1)
    df['TTF'] = (df['top3_mean'] / (df['bottom3_mean'] + 1e-6)) * (df['l2_norm'] / (df['sum'] + 1e-6))
    df.drop(columns=['top3_mean', 'bottom3_mean'], inplace=True)
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)

    return df

def predict_approach2_batch(X_scaled, models):
    device = models['device']
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    models['approach2'].eval()
    with torch.no_grad():
        preds = models['approach2'](X_tensor).cpu().numpy().flatten()
    return preds

def predict_approach3_batch(X_scaled, models):
    resnet_pred = predict_approach2_batch(X_scaled, models)
    cat_pred = models['approach3']['catboost'].predict(X_scaled)
    xgb_pred = models['approach3']['xgboost'].predict(X_scaled)
    stack_input = np.column_stack([resnet_pred, cat_pred, xgb_pred])
    return models['approach3']['ridge_meta'].predict(stack_input)

def predict_approach4_batch(X_scaled, models):
    resnet_pred = predict_approach2_batch(X_scaled, models)
    approach3_pred = predict_approach3_batch(X_scaled, models)
    return 0.7 * approach3_pred + 0.3 * resnet_pred

# Fill missing values
print("Filling missing values...")
for col in train_df.columns:
    median_val = train_df[col].median()
    if pd.isna(median_val):
        train_df[col] = train_df[col].fillna(0.5)
    else:
        train_df[col] = train_df[col].fillna(median_val)

# Feature engineering
print("Applying feature engineering...")
train_eng = advanced_features(train_df)
exclude = ['id', 'FloodProbability']
feature_cols = [c for c in train_eng.columns if c not in exclude]

# Baseline prediction
print("Computing baseline predictions...")
baseline_array = train_eng[feature_cols].values
baseline_scaled = scaler.transform(baseline_array)
baseline_pred = predict_approach4_batch(baseline_scaled, models_dict)

# Sensitivity analysis
print("Computing sensitivity for each feature...")
sensitivities = {}
for i, feature in enumerate(base_features):
    print(f"  [{i+1}/{len(base_features)}] {feature}...", end=' ', flush=True)
    
    perturbed_df = train_df.copy()
    perturbed_df[feature] = np.clip(perturbed_df[feature] + 0.1, 0.0, 1.0)

    perturbed_eng = advanced_features(perturbed_df)
    perturbed_array = perturbed_eng[feature_cols].values
    perturbed_scaled = scaler.transform(perturbed_array)
    perturbed_pred = predict_approach4_batch(perturbed_scaled, models_dict)

    impact = float(np.mean(np.abs(perturbed_pred - baseline_pred)))
    sensitivities[feature] = impact
    print(f"{impact:.6f}")

# Save results
results_df = pd.DataFrame({
    'Feature': list(sensitivities.keys()),
    'Impact': list(sensitivities.values())
}).sort_values('Impact', ascending=False)

results_df.to_csv('train_sensitivity_approach4.csv', index=False)
joblib.dump(sensitivities, 'train_sensitivity_approach4.pkl')

print(f"\n✅ Saved sensitivity results!")
print(f"\nTop 10 Features by Sensitivity (Approach 4, 1.4M rows):")
print(results_df.head(10).to_string(index=False))
print(f"\nFiles saved:")
print("  - train_sensitivity_approach4.csv")
print("  - train_sensitivity_approach4.pkl")
