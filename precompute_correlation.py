"""
Pre-compute correlation matrix from train.csv and save it for fast loading
Run this script once: python precompute_correlation.py
"""

import pandas as pd
import numpy as np

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

print("Computing correlation matrix...")
corr_matrix = train_df.corr()
print("Correlation matrix computed!")

# Save as CSV
corr_matrix.to_csv('train_correlation_matrix.csv')
print("✅ Saved to: train_correlation_matrix.csv")

# Also save as pickle for faster loading
import joblib
joblib.dump(corr_matrix, 'train_correlation_matrix.pkl')
print("✅ Saved to: train_correlation_matrix.pkl")

print(f"\nCorrelation matrix shape: {corr_matrix.shape}")
print("Done! Add 'train_correlation_matrix.pkl' to .gitignore if not uploading to GitHub.")
