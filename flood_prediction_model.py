"""
flood_prediction_model.py - Train and save flood prediction models

This script trains 4 different approaches for flood probability prediction:
1. First Approach : Blending (LightGBM + CatBoost + XGBoost + HistGradientBoosting) + Ridge Meta-learner
2. Second Approach: ResNet neural network
3. Third Approach : Stacking (ResNet + CatBoost + XGBoost) + Ridge Meta-learner  
4. Final Approach : Ensemble voting (90% Third + 10% Second)

Run this script FIRST to create the model files: python flood_prediction_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import skew, kurtosis, entropy
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

print("=" * 70)
print("TRAINING FLOOD PROBABILITY PREDICTION MODELS (4 Approaches)")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading dataset...")

# NOTE: Make sure train.csv and test.csv are in the project directory
# If using Kaggle data, download them first

try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_ids = test_df['id'].copy() if 'id' in test_df.columns else None
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
except FileNotFoundError as e:
    print(f"   ❌ Error: {e}")
    print("   Please ensure train.csv and test.csv are in the project directory")
    raise

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. Data preprocessing...")

# Remove duplicates
train_df = train_df.drop_duplicates().reset_index(drop=True)
print(f"   Duplicates removed. Final train shape: {train_df.shape}")

# Handle missing values
feature_cols = [c for c in train_df.columns if c not in ['id', 'FloodProbability']]
for col in feature_cols:
    if train_df[col].isnull().sum() > 0:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

print(f"   Missing values handled")

# ============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n3. Advanced feature engineering...")

def advanced_features(df):
    """Create advanced engineered features from base features"""
    df = df.copy()
    base_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
        'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
        'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
    ]

    data = df[base_features].values

    # Row-wise Statistics
    df['sum'] = data.sum(axis=1)
    df['mean'] = data.mean(axis=1)
    df['std'] = data.std(axis=1)
    df['skew'] = skew(data, axis=1)
    df['kurt'] = kurtosis(data, axis=1)
    df['max'] = data.max(axis=1)
    df['min'] = data.min(axis=1)
    df['range'] = df['max'] - df['min']
    df['sum_sq'] = np.sum(data**2, axis=1)
    df['ptp'] = np.ptp(data, axis=1)

    # Quantiles
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        df[f'q{int(q*100):02d}'] = np.quantile(data, q, axis=1)

    # Dispersion & Robustness
    df['iqr'] = df['q75'] - df['q25']
    df['cv'] = df['std'] / (df['mean'] + 1e-6)
    df['mad'] = np.mean(np.abs(data - df['mean'].values[:, None]), axis=1)

    # Norm-based & Distribution
    df['l1_norm'] = np.sum(np.abs(data), axis=1)
    df['l2_norm'] = np.linalg.norm(data, axis=1)
    df['entropy'] = entropy(np.abs(data) + 1e-9, axis=1)

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

    return df

train_df = advanced_features(train_df)
test_df = advanced_features(test_df)
print(f"   Feature engineering complete. Total features: {train_df.shape[1]}")

# ============================================================================
# 4. DATA SCALING AND TRAIN-VAL SPLIT
# ============================================================================
print("\n4. Scaling features and splitting data...")

exclude = ['id', 'FloodProbability']
feature_cols = [c for c in train_df.columns if c not in exclude]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[feature_cols])
y = train_df['FloodProbability'].values
X_test_scaled = scaler.transform(test_df[feature_cols])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_scaled.shape}")

# Configuration for GPU device (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

# ============================================================================
# 5. DEFINE RESNET ARCHITECTURE (for Approach 2 and 3)
# ============================================================================
print("\n5. Defining ResNet architecture...")

class ResidualBlock(nn.Module):
    """Residual block for deep tabular learning"""
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
    """ResNet for tabular flood prediction"""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        for block in self.blocks:
            out = block(out)
        out = torch.sigmoid(self.output_layer(out))  # Output between 0 and 1
        return out

# ============================================================================
# 6. APPROACH 1: BLENDING + STACKING WITH RIDGE META-LEARNER
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 1: LightGBM + CatBoost + XGBoost + HistGB -> Ridge")
print("=" * 70)

def train_approach1():
    print("\nTraining Approach 1 models...")
    
    # LightGBM
    print("  • Training LightGBM...")
    lgb_model = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        subsample=0.8, verbose=-1, random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_oof = lgb_model.predict(X_val)
    lgb_test = lgb_model.predict(X_test_scaled)
    
    # CatBoost
    print("  • Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05, 
        verbose=0, random_state=42
    )
    cat_model.fit(X_train, y_train)
    cat_oof = cat_model.predict(X_val)
    cat_test = cat_model.predict(X_test_scaled)
    
    # XGBoost
    print("  • Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        subsample=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_oof = xgb_model.predict(X_val)
    xgb_test = xgb_model.predict(X_test_scaled)
    
    # HistGradientBoosting
    print("  • Training HistGradientBoosting...")
    hgb_model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, random_state=42
    )
    hgb_model.fit(X_train, y_train)
    hgb_oof = hgb_model.predict(X_val)
    hgb_test = hgb_model.predict(X_test_scaled)
    
    # Stack predictions
    stack_train = np.column_stack([lgb_oof, cat_oof, xgb_oof, hgb_oof])
    stack_test = np.column_stack([lgb_test, cat_test, xgb_test, hgb_test])
    
    # Train Ridge meta-learner
    ridge_meta = RidgeCV(alphas=np.logspace(-3, 1, 50), cv=5)
    ridge_meta.fit(stack_train, y_val)
    
    predictions = ridge_meta.predict(stack_test)
    
    models = {
        'lgb': lgb_model,
        'cat': cat_model,
        'xgb': xgb_model,
        'hgb': hgb_model,
        'ridge_meta': ridge_meta,
        'predictions': predictions
    }
    
    rmse = np.sqrt(mean_squared_error(y_val, ridge_meta.predict(stack_train)))
    print(f"  Approach 1 Val RMSE: {rmse:.6f}")
    
    return models, predictions, rmse

approach1_models, approach1_test_pred, approach1_rmse = train_approach1()
joblib.dump(approach1_models, 'approach1_models.pkl')
print("  ✓ Approach 1 models saved")

# ============================================================================
# 7. APPROACH 2: RESNET NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 2: ResNet Tabular Neural Network")
print("=" * 70)

def train_approach2():
    print("\nTraining Approach 2 model...")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ResNetTabular(input_dim=X_train_tensor.shape[1], hidden_dim=256, num_blocks=4).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 25
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    # Predict on test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    torch.save(model.state_dict(), 'approach2_resnet.pt')
    print(f"  ✓ Approach 2 model saved")
    print(f"  Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    return model, test_pred

approach2_model, approach2_test_pred = train_approach2()

# ============================================================================
# 8. APPROACH 3: STACKING (ResNet + CatBoost + XGBoost) + RIDGE
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 3: Stacking (ResNet + CatBoost + XGBoost) -> Ridge")
print("=" * 70)

def train_approach3():
    print("\nTraining Approach 3 models...")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # Use the trained ResNet from Approach 2
    approach2_model.eval()
    with torch.no_grad():
        resnet_train = approach2_model(X_train_tensor).cpu().numpy().flatten()
        resnet_val = approach2_model(X_val_tensor).cpu().numpy().flatten()
        resnet_test = approach2_model(X_test_tensor).cpu().numpy().flatten()
    
    # Train CatBoost
    print("  • Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05, verbose=0, random_state=42
    )
    cat_model.fit(X_train, y_train)
    cat_train = cat_model.predict(X_train)
    cat_val = cat_model.predict(X_val)
    cat_test = cat_model.predict(X_test_scaled)
    
    # Train XGBoost
    print("  • Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        subsample=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_train = xgb_model.predict(X_train)
    xgb_val = xgb_model.predict(X_val)
    xgb_test = xgb_model.predict(X_test_scaled)
    
    # Stack predictions
    stack_train = np.vstack([resnet_train, cat_train, xgb_train]).T
    stack_val = np.vstack([resnet_val, cat_val, xgb_val]).T
    stack_test = np.vstack([resnet_test, cat_test, xgb_test]).T
    
    # Train Ridge meta-learner
    ridge_meta = Ridge(alpha=1.0)
    ridge_meta.fit(stack_train, y_train)
    
    val_pred = ridge_meta.predict(stack_val)
    test_pred = ridge_meta.predict(stack_test)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  Approach 3 Val RMSE: {rmse:.6f}")
    
    models = {
        'catboost': cat_model,
        'xgboost': xgb_model,
        'ridge_meta': ridge_meta,
        'predictions': test_pred
    }
    
    return models, test_pred, rmse

approach3_models, approach3_test_pred, approach3_rmse = train_approach3()
joblib.dump(approach3_models, 'approach3_models.pkl')
print("  ✓ Approach 3 models saved")

# ============================================================================
# 9. APPROACH 4: FINAL ENSEMBLE (VOTING)
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 4: Final Ensemble (70% Approach3 + 30% Approach2)")
print("=" * 70)

approach4_test_pred = 0.7 * approach3_test_pred + 0.3 * approach2_test_pred
approach4_test_pred = np.clip(approach4_test_pred, 0, 1)

print(f"Test predictions range: [{approach4_test_pred.min():.4f}, {approach4_test_pred.max():.4f}]")

# ============================================================================
# 10. SAVE ALL MODELS AND METADATA
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS AND METADATA")
print("=" * 70)

# Save scaler
joblib.dump(scaler, 'flood_scaler.pkl')
print("✓ Scaler saved")

# Save metadata
model_metadata = {
    'feature_names': feature_cols,
    'n_features': len(feature_cols),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'approaches': {
        'approach1': {'rmse': approach1_rmse},
        'approach2': {'type': 'ResNet'},
        'approach3': {'rmse': approach3_rmse},
        'approach4': {'type': 'Voting (70% A3 + 30% A2)', 'weights': {'approach3': 0.7, 'approach2': 0.3}}
    }
}
joblib.dump(model_metadata, 'flood_model_metadata.pkl')
print("✓ Model metadata saved")

# Save all predictions for reference
all_predictions = {
    'approach1': approach1_test_pred,
    'approach2': approach2_test_pred,
    'approach3': approach3_test_pred,
    'approach4': approach4_test_pred
}
joblib.dump(all_predictions, 'all_predictions.pkl')
print("✓ All predictions saved")

# Summary
print("\n" + "=" * 70)
print("MODEL TRAINING SUMMARY")
print("=" * 70)
print(f"Approach 1 (Blending + Ridge)     Val RMSE: {approach1_rmse:.6f}")
print(f"Approach 2 (ResNet)                Type: Neural Network")
print(f"Approach 3 (Stacking + Ridge)     Val RMSE: {approach3_rmse:.6f}")
print(f"Approach 4 (Voting 90/10)         Final Ensemble")

print("\n" + "=" * 70)
print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print("\nModel files created:")
print("  • approach1_models.pkl")
print("  • approach2_resnet.pt")
print("  • approach3_models.pkl")
print("  • flood_scaler.pkl")
print("  • flood_model_metadata.pkl")
print("  • all_predictions.pkl")
print("\nNext step: Run the Streamlit app with:")
print("  streamlit run app_flood.py")
print("=" * 70)

