"""
Feature Enhancements for RiverGuard
- Feature Dictionary with detailed descriptions
- Risk Level Classification
- Confidence Intervals estimation
- Feature Engineering Impact analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# 1. FEATURE DICTIONARY - Detailed Descriptions
# ============================================================================
FEATURE_DICTIONARY = {
    # Input Features (20 base features)
    'MonsoonIntensity': {
        'description': 'Intensity of monsoon rainfall patterns',
        'impact': 'Higher intensity increases flood risk by excessive precipitation',
        'range': '[0, 1]', 'unit': 'Normalized Index'
    },
    'TopographyDrainage': {
        'description': 'Terrain drainage efficiency and slope characteristics',
        'impact': 'Poor drainage increases water accumulation and flood risk',
        'range': '[0, 1]', 'unit': 'Efficiency Score'
    },
    'RiverManagement': {
        'description': 'Quality of river channel maintenance and management practices',
        'impact': 'Poor management increases clogging and overflow risk',
        'range': '[0, 1]', 'unit': 'Quality Score'
    },
    'Deforestation': {
        'description': 'Level of forest cover loss in the watershed area',
        'impact': 'Higher deforestation → reduced water absorption → higher flood risk',
        'range': '[0, 1]', 'unit': 'Loss Index'
    },
    'Urbanization': {
        'description': 'Extent of urban development and concrete coverage',
        'impact': 'More concrete reduces infiltration, increases runoff and floods',
        'range': '[0, 1]', 'unit': 'Urban Index'
    },
    'ClimateChange': {
        'description': 'Long-term climate change impacts on precipitation patterns',
        'impact': 'Increases extreme weather events and rainfall variability',
        'range': '[0, 1]', 'unit': 'Change Index'
    },
    'DamsQuality': {
        'description': 'Structural quality and maintenance of dams and reservoirs',
        'impact': 'Poor dams may fail or overflow, causing downstream floods',
        'range': '[0, 1]', 'unit': 'Quality Score'
    },
    'Siltation': {
        'description': 'Accumulation of sediment in river channels and reservoirs',
        'impact': 'Reduces water capacity, increases overflow and flooding risk',
        'range': '[0, 1]', 'unit': 'Siltation Index'
    },
    'AgriculturalPractices': {
        'description': 'Sustainability of farming practices (soil health, irrigation)',
        'impact': 'Poor practices increase runoff and reduce water infiltration',
        'range': '[0, 1]', 'unit': 'Sustainability Score'
    },
    'Encroachments': {
        'description': 'Encroachment on natural floodplains and waterways',
        'impact': 'Reduces natural flood corridors, increases flood magnitude',
        'range': '[0, 1]', 'unit': 'Encroachment Index'
    },
    'IneffectiveDisasterPreparedness': {
        'description': 'Preparedness level for flood events (warning systems, evacuation)',
        'impact': 'Poor preparedness means higher impact when floods occur',
        'range': '[0, 1]', 'unit': 'Preparedness Score'
    },
    'DrainageSystems': {
        'description': 'Efficiency of urban and rural drainage infrastructure',
        'impact': 'Poor drainage systems increase waterlogging and flooding',
        'range': '[0, 1]', 'unit': 'Efficiency Score'
    },
    'CoastalVulnerability': {
        'description': 'Vulnerability of coastal areas to storm surge and flooding',
        'impact': 'Coastal areas more susceptible to flooding from ocean/storms',
        'range': '[0, 1]', 'unit': 'Vulnerability Index'
    },
    'Landslides': {
        'description': 'Risk and frequency of landslides in the region',
        'impact': 'Landslides block rivers and create natural dams causing floods',
        'range': '[0, 1]', 'unit': 'Risk Index'
    },
    'Watersheds': {
        'description': 'Watershed health and protection status',
        'impact': 'Degraded watersheds have poor water retention, increase runoff',
        'range': '[0, 1]', 'unit': 'Health Score'
    },
    'DeterioratingInfrastructure': {
        'description': 'Condition of flood protection infrastructure (levees, barriers)',
        'impact': 'Deteriorating barriers fail to contain flood waters',
        'range': '[0, 1]', 'unit': 'Deterioration Index'
    },
    'PopulationScore': {
        'description': 'Population density in flood-prone areas',
        'impact': 'More people in risk areas increases vulnerability',
        'range': '[0, 1]', 'unit': 'Density Score'
    },
    'WetlandLoss': {
        'description': 'Loss of wetland ecosystems that naturally absorb water',
        'impact': 'Wetland loss removes natural flood buffers',
        'range': '[0, 1]', 'unit': 'Loss Index'
    },
    'InadequatePlanning': {
        'description': 'Urban planning quality and consideration of flood risks',
        'impact': 'Poor planning builds in risky areas, inadequate infrastructure',
        'range': '[0, 1]', 'unit': 'Planning Score'
    },
    'PoliticalFactors': {
        'description': 'Political will and governance for flood mitigation',
        'impact': 'Weak governance = poor flood management policies and enforcement',
        'range': '[0, 1]', 'unit': 'Governance Score'
    }
}

# Engineered Features Groups
ENGINEERED_FEATURES_GROUPS = {
    'Statistical': ['sum', 'mean', 'std', 'skew', 'kurt', 'max', 'min', 'range', 'sum_sq', 'ptp'],
    'Quantiles': ['q01', 'q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99'],
    'Dispersion': ['iqr', 'cv', 'mad'],
    'Norms & Entropy': ['l1_norm', 'l2_norm', 'entropy'],
    'Sorted Features': ['sort_0', 'sort_1', 'sort_2', 'sort_3', 'sort_4', 'sort_5', 'sort_6', 'sort_7', 'sort_8', 'sort_9', 
                       'sort_10', 'sort_11', 'sort_12', 'sort_13', 'sort_14', 'sort_15', 'sort_16', 'sort_17', 'sort_18', 'sort_19'],
    'Custom Coefficients': ['CCP', 'TTF']
}

# ============================================================================
# 2. RISK LEVEL CLASSIFICATION
# ============================================================================
def classify_flood_risk(probability):
    """
    Classify flood risk into 4 levels based on probability
    
    Args:
        probability: Float between 0 and 1
    
    Returns:
        dict with level, emoji, color, description
    """
    if probability < 0.25:
        return {
            'level': '🟢 LOW RISK',
            'emoji': '🟢',
            'color': '#388E3C',
            'bg_color': '#E8F5E9',
            'description': 'Low flood probability. Normal conditions expected.'
        }
    elif probability < 0.5:
        return {
            'level': '🟡 MEDIUM RISK',
            'emoji': '🟡',
            'color': '#F57C00',
            'bg_color': '#FFF3E0',
            'description': 'Moderate flood probability. Monitor weather conditions.'
        }
    elif probability < 0.75:
        return {
            'level': '🔴 HIGH RISK',
            'emoji': '🔴',
            'color': '#D32F2F',
            'bg_color': '#FFEBEE',
            'description': 'High flood probability. Prepare for potential flooding.'
        }
    else:
        return {
            'level': '🟥 CRITICAL RISK',
            'emoji': '🟥',
            'color': '#B71C1C',
            'bg_color': '#DD2C00',
            'description': 'Critical flood risk. Emergency response needed immediately.'
        }

# ============================================================================
# 3. PREDICTION CONFIDENCE INTERVALS
# ============================================================================
def estimate_confidence_interval(predictions, confidence=0.95):
    """
    Estimate confidence intervals for predictions using bootstrap
    
    Args:
        predictions: Array of predictions
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        dict with lower, mean, upper bounds
    """
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # Using normal distribution approximation
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99%
    margin_of_error = z_score * (std_pred / np.sqrt(len(predictions)))
    
    return {
        'lower': max(0, mean_pred - margin_of_error),
        'mean': mean_pred,
        'upper': min(1, mean_pred + margin_of_error),
        'margin': margin_of_error
    }

# ============================================================================
# 4. FEATURE ENGINEERING IMPACT ANALYSIS
# ============================================================================
def create_feature_engineering_comparison(base_features_importance, engineered_features_importance):
    """
    Create visualization comparing base vs engineered feature importance
    
    Args:
        base_features_importance: Dict of base feature importances
        engineered_features_importance: Dict of engineered feature importances
    
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    # Add base features
    fig.add_trace(go.Bar(
        x=list(base_features_importance.keys())[:10],
        y=list(base_features_importance.values())[:10],
        name='Base Features',
        marker_color='rgba(21, 101, 192, 0.7)',
        showlegend=True
    ))
    
    # Add engineered features
    fig.add_trace(go.Bar(
        x=list(engineered_features_importance.keys())[:10],
        y=list(engineered_features_importance.values())[:10],
        name='Engineered Features',
        marker_color='rgba(244, 67, 54, 0.7)',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Feature Engineering Impact: Base vs Engineered Features",
        xaxis_title="Features",
        yaxis_title="Importance / Sensitivity",
        barmode='group',
        hovermode='x unified',
        height=500
    )
    
    return fig

# ============================================================================
# 5. TRAINING DATA STRATIFICATION ANALYSIS
# ============================================================================
def analyze_class_distribution(train_df, target_col='FloodProbability', n_bins=10):
    """
    Analyze distribution of target variable across different ranges
    
    Args:
        train_df: Training dataframe
        target_col: Name of target column
        n_bins: Number of bins for stratification
    
    Returns:
        DataFrame with stratification analysis
    """
    if target_col not in train_df.columns:
        return None
    
    target = train_df[target_col]
    
    # Create bins
    bins = np.linspace(target.min(), target.max(), n_bins + 1)
    labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
    
    binned_target = pd.cut(target, bins=bins, labels=labels, include_lowest=True)
    
    stratification = binned_target.value_counts().sort_index()
    stratification_pct = (stratification / len(target) * 100).round(2)
    
    analysis_df = pd.DataFrame({
        'Probability Range': stratification.index,
        'Count': stratification.values,
        'Percentage': stratification_pct.values
    })
    
    return analysis_df

# ============================================================================
# 6. FEATURE-TARGET CORRELATION ANALYSIS
# ============================================================================
def compute_feature_target_correlation(df, base_features, target_col='FloodProbability'):
    """
    Compute correlation between each feature and target
    
    Args:
        df: DataFrame with features and target
        base_features: List of feature names
        target_col: Name of target column
    
    Returns:
        DataFrame sorted by absolute correlation
    """
    correlations = {}
    
    for feature in base_features:
        if feature in df.columns:
            corr = df[feature].corr(df[target_col])
            correlations[feature] = corr
    
    corr_df = pd.DataFrame(list(correlations.items()), 
                           columns=['Feature', 'Correlation'])
    # Sort by absolute value of correlation but keep only Correlation column
    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
    
    return corr_df

# ============================================================================
# 7. SHAP PREPARATION (requires SHAP library)
# ============================================================================
def check_shap_availability():
    """Check if SHAP library is available"""
    try:
        import shap
        return True, shap
    except ImportError:
        return False, None

# ============================================================================
# 8. REGIONAL FLOOD RISK MAPPING
# ============================================================================
def check_geographic_data(train_df):
    """
    Check if geographic data columns exist in dataset
    
    Args:
        train_df: Training dataframe
    
    Returns:
        dict with geographic columns found
    """
    geographic_keywords = [
        'latitude', 'longitude', 'lat', 'lon', 'region', 'district', 'city',
        'state', 'province', 'area', 'zone', 'coordinates', 'location'
    ]
    
    found_columns = {}
    df_columns_lower = {col.lower(): col for col in train_df.columns}
    
    for keyword in geographic_keywords:
        if keyword in df_columns_lower:
            found_columns[keyword] = df_columns_lower[keyword]
    
    return found_columns

def create_risk_map_placeholder():
    """Create placeholder for regional risk mapping"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="Geographic data (latitude/longitude) not found in training dataset.<br>" +
             "This feature requires geographic coordinates for regional flood risk mapping.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14),
        align="center"
    )
    
    fig.update_layout(
        title="Regional Flood Risk Mapping",
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    return fig
