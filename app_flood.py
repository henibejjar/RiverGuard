"""
app_flood.py - Advanced Flood Prediction Web Interface

Enhanced features:
✓ Support for missing values (NaN)
✓ Auto-imputation with median values
✓ Interactive visualizations & plots
✓ Feature importance analysis
✓ Sensitivity analysis
✓ Batch predictions with export
✓ Data normalization explanation
✓ Risk assessment dashboard
✓ Model explainability tools
✓ Input validation & statistics

Run with: streamlit run app_flood.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
import warnings
import os

# Import feature enhancements
try:
    from feature_enhancements import (
        FEATURE_DICTIONARY, ENGINEERED_FEATURES_GROUPS,
        estimate_confidence_interval,
        analyze_class_distribution, compute_feature_target_correlation,
        check_shap_availability, check_geographic_data, create_risk_map_placeholder
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# DATA HELPERS
# ============================================================================
@st.cache_data(show_spinner=False)
def load_train_data(path, usecols=None):
    return pd.read_csv(path, usecols=usecols)

@st.cache_data(show_spinner=False)
def load_correlation_matrix():
    """Load pre-computed correlation matrix or compute if not available"""
@st.cache_data(show_spinner=False)
def load_correlation_matrix():
    """Load pre-computed correlation matrix or compute if not available"""
    # Try loading pre-computed matrix (fast)
    if os.path.exists('train_correlation_matrix.pkl'):
        import joblib
        return joblib.load('train_correlation_matrix.pkl'), True
    elif os.path.exists('train_correlation_matrix.csv'):
        return pd.read_csv('train_correlation_matrix.csv', index_col=0), True
    return None, False

@st.cache_data(show_spinner=False)
def load_sensitivity_results():
    """Load pre-computed feature sensitivity on 1.4M training rows"""
    if os.path.exists('train_sensitivity_approach4.pkl'):
        return joblib.load('train_sensitivity_approach4.pkl'), True
    elif os.path.exists('train_sensitivity_approach4.csv'):
        df = pd.read_csv('train_sensitivity_approach4.csv', index_col=0)
        return df.to_dict('list'), True
    return None, False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🌊 RiverGuard - Flood Probability Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: #FFFFFF;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: bold;
    }
    
    .main-title p {
        color: #FFFFFF;
    }
    
    .prediction-box {
        padding: 25px;
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        border-left: 5px solid #FFD700;
        border-radius: 8px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        color: #FFFFFF;
    }
    
    .risk-high {
        padding: 15px;
        background-color: #FFEBEE;
        border-left: 4px solid #D32F2F;
        border-radius: 5px;
        color: #D32F2F;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .risk-medium {
        padding: 15px;
        background-color: #FFF3E0;
        border-left: 4px solid #F57C00;
        border-radius: 5px;
        color: #F57C00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .risk-low {
        padding: 15px;
        background-color: #E8F5E9;
        border-left: 4px solid #388E3C;
        border-radius: 5px;
        color: #388E3C;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .info-box {
        padding: 15px;
        background-color: #E1F5FE;
        border-left: 4px solid #0288D1;
        border-radius: 5px;
        color: #01579B;
    }
    
    .warning-box {
        padding: 15px;
        background-color: #FFF8E1;
        border-left: 4px solid #FBC02D;
        border-radius: 5px;
        color: #F57F17;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# RESNET ARCHITECTURE
# ============================================================================
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

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('flood_scaler.pkl')
        metadata = joblib.load('flood_model_metadata.pkl')
        approach1_models = joblib.load('approach1_models.pkl')
        approach3_models = joblib.load('approach3_models.pkl')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet_model = ResNetTabular(input_dim=metadata['n_features'], hidden_dim=256, num_blocks=4)
        resnet_state = torch.load('approach2_resnet.pt', map_location=device, weights_only=False)
        resnet_model.load_state_dict(resnet_state)
        resnet_model.to(device)
        resnet_model.eval()
        
        return {
            'scaler': scaler,
            'metadata': metadata,
            'approach1': approach1_models,
            'approach2': resnet_model,
            'approach3': approach3_models,
            'device': device
        }, None
    except FileNotFoundError as e:
        return None, f"❌ Models not found: {str(e)}. Run 'python flood_prediction_model.py' first."

models_dict, load_error = load_models()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
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

    # CRITICAL: Fill NaN values BEFORE feature engineering
    for col in base_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0.5)
    
    data = df[base_features].values

    # Row-wise Statistics
    df['sum'] = np.nansum(data, axis=1)
    df['mean'] = np.nanmean(data, axis=1)
    df['std'] = np.nanstd(data, axis=1)
    df['std'] = df['std'].fillna(0)  # Fill std NaN with 0
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

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_approach1_batch(X_scaled, models):
    lgb_pred = models['approach1']['lgb'].predict(X_scaled)
    cat_pred = models['approach1']['cat'].predict(X_scaled)
    xgb_pred = models['approach1']['xgb'].predict(X_scaled)
    hgb_pred = models['approach1']['hgb'].predict(X_scaled)
    stack_input = np.column_stack([lgb_pred, cat_pred, xgb_pred, hgb_pred])
    return models['approach1']['ridge_meta'].predict(stack_input)

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

def predict_with_approach(approach_num, X_scaled, models):
    if approach_num == 1:
        return predict_approach1_batch(X_scaled, models)
    if approach_num == 2:
        return predict_approach2_batch(X_scaled, models)
    if approach_num == 3:
        return predict_approach3_batch(X_scaled, models)
    if approach_num == 4:
        return predict_approach4_batch(X_scaled, models)
    raise ValueError(f"Unknown approach: {approach_num}")

def risk_label(probability):
    """Get risk level label"""
    if probability >= 0.7:
        return "🔴 High Risk"
    if probability >= 0.4:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

def risk_color(probability):
    """Get risk level color"""
    if probability >= 0.7:
        return "#D32F2F"
    if probability >= 0.4:
        return "#F57C00"
    return "#388E3C"

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
    <div class="main-title">
        🌊 RiverGuard - Advanced Flood Probability Predictor
        <p style="font-size: 0.8rem; margin-top: 10px;">Powered by Ensemble Machine Learning & Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

# Check if models loaded
if load_error:
    st.error(load_error)
    st.stop()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    app_mode = st.radio(
        "Select Mode:",
        ["🎯 Quick Prediction", "🔬 Advanced Analysis", "📊 Batch Processing", "📚 Help & Info"],
        help="Choose how you want to interact with the model"
    )
    
    approach_num = st.selectbox(
        "Select Prediction Approach:",
        [4, 3, 1, 2],
        format_func=lambda x: {
            1: "Approach 1: Blending + Ridge",
            2: "Approach 2: ResNet NN",
            3: "Approach 3: Stacking",
            4: "Approach 4: Final Ensemble (Recommended)"
        }[x],
        help="Approach 4 combines Approach 3 (70%) + Approach 2 (30%) for best results"
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: Features should be in [0, 1] range. Leave NaN for missing data.")

# ============================================================================
# QUICK PREDICTION MODE
# ============================================================================
if app_mode == "🎯 Quick Prediction":
    st.header("Single Prediction")
    
    base_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
        'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
        'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
    ]
    
    # Preset scenarios
    st.markdown("### ⚡ Quick Presets")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📍 Low Risk", width='stretch', key="preset_low"):
            for feat in base_features:
                st.session_state[feat] = 0.2
            st.rerun()
    with col2:
        if st.button("📍 Balanced", width='stretch', key="preset_balanced"):
            for feat in base_features:
                st.session_state[feat] = 0.5
            st.rerun()
    with col3:
        if st.button("📍 High Risk", width='stretch', key="preset_high"):
            for feat in base_features:
                st.session_state[feat] = 0.8
            st.rerun()
    with col4:
        if st.button("🔄 Reset All", width='stretch', key="preset_reset"):
            for feat in base_features:
                st.session_state[feat] = 0.5
            st.rerun()
    
    st.markdown("---")
    
    # Input mode selector
    st.markdown("### 🎛️ Input Mode Selection")
    input_mode = st.radio(
        "Choose how to input feature values:",
        ["🎚️ Sliders (Drag & Drop)", "⌨️ Text Input (Type Values)"],
        horizontal=True,
        help="Sliders: visual and interactive | Text: precise values"
    )
    
    st.markdown("---")
    
    input_values = {}
    cols = st.columns(4)
    
    if input_mode == "🎚️ Sliders (Drag & Drop)":
        st.subheader("📊 Drag Sliders to Set Feature Values [0-1]:")
        st.info("💡 All values in this mode are between 0 and 1")
        
        for idx, feature in enumerate(base_features):
            if feature not in st.session_state:
                st.session_state[feature] = 0.5
            
            with cols[idx % 4]:
                # Slider input
                value = st.slider(
                    feature,
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state[feature] if isinstance(st.session_state[feature], (int, float)) else 0.5,
                    step=0.05,
                    key=f"{feature}_slider",
                    help=f"Drag to set {feature}"
                )
                input_values[feature] = value
                st.session_state[feature] = value
    
    else:  # Text Input Mode
        st.subheader("⌨️ Enter Feature Values [0-1] or Use *, -, _ for Missing:")
        st.info("💡 Enter numbers between 0-1, or *, -, _ for missing values (will be auto-imputed)")
        
        for idx, feature in enumerate(base_features):
            if feature not in st.session_state:
                st.session_state[feature] = "0.5"
            
            with cols[idx % 4]:
                val = st.text_input(
                    feature,
                    value=st.session_state[feature] if isinstance(st.session_state[feature], str) else str(st.session_state[feature]),
                    key=f"{feature}_input",
                    help="0-1 or *, -, _ for missing"
                )
                
                # Parse input
                if val.strip() in ['*', '-', '_', '']:
                    input_values[feature] = np.nan
                    st.session_state[feature] = val
                else:
                    try:
                        num_val = float(val)
                        # Clamp to [0, 1]
                        num_val = np.clip(num_val, 0.0, 1.0)
                        input_values[feature] = num_val
                        st.session_state[feature] = str(num_val)
                    except (ValueError, TypeError):
                        st.warning(f"⚠️ Invalid value for {feature}, using 0.5")
                        input_values[feature] = 0.5
                        st.session_state[feature] = "0.5"
    
    st.markdown("---")
    
    # Input summary
    col1, col2 = st.columns(2)
    with col1:
        input_df = pd.DataFrame({
            'Feature': list(input_values.keys()),
            'Value': list(input_values.values())
        })
        st.subheader("Input Summary")
        
        # Format display for NaN values
        display_df = input_df.copy()
        display_df['Value'] = display_df['Value'].apply(
            lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"
        )
        st.dataframe(display_df, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("Statistics")
        # Filter out NaN values for statistics
        valid_values = [v for v in input_values.values() if not pd.isna(v)]
        
        if len(valid_values) > 0:
            stats_cols = st.columns(2)
            with stats_cols[0]:
                st.metric("Mean", f"{np.mean(valid_values):.3f}")
                st.metric("Min", f"{np.min(valid_values):.3f}")
            with stats_cols[1]:
                st.metric("Max", f"{np.max(valid_values):.3f}")
                st.metric("Std Dev", f"{np.std(valid_values):.3f}")
            
            if len(valid_values) < len(input_values):
                st.info(f"ℹ️ {len(input_values) - len(valid_values)} missing value(s) will be auto-imputed")
        else:
            st.warning("⚠️ All values are missing. Using default median values.")
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Make Prediction", type="primary", width='stretch')
    
    if predict_button:
        # Prepare data
        input_df_raw = pd.DataFrame([input_values])
        input_df_engineered = advanced_features(input_df_raw)
        
        exclude = ['id', 'FloodProbability']
        feature_cols = [c for c in input_df_engineered.columns if c not in exclude]
        input_array = input_df_engineered[feature_cols].values
        
        if not np.isfinite(input_array).all():
            st.error("❌ Non-finite values in engineered features")
            st.stop()
        
        scaler = models_dict['scaler']
        X_scaled = scaler.transform(input_array)
        
        # Prediction
        approach_names = {
            1: "Approach 1: Blending + Ridge",
            2: "Approach 2: ResNet",
            3: "Approach 3: Stacking + Ridge",
            4: "Approach 4: Final Ensemble"
        }
        
        predictions = {
            1: predict_approach1_batch(X_scaled, models_dict)[0],
            2: predict_approach2_batch(X_scaled, models_dict)[0],
            3: predict_approach3_batch(X_scaled, models_dict)[0],
            4: predict_approach4_batch(X_scaled, models_dict)[0]
        }
        
        prediction = np.clip(predictions[approach_num], 0, 1)
        approach_name = approach_names[approach_num]
        
        # Calculate confidence intervals from all 4 approaches
        all_predictions = np.array([np.clip(p, 0, 1) for p in predictions.values()])
        ci_results = estimate_confidence_interval(all_predictions, confidence=0.95) if ENHANCEMENTS_AVAILABLE else None
        
        # Display main prediction with confidence interval
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
                <div class="prediction-box">
                    <div>Flood Probability: {prediction*100:.2f}%</div>
                    <div style="font-size: 1.2rem; margin-top: 10px;">{risk_label(prediction)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if ci_results:
                st.metric(
                    "95% Confidence Interval",
                    f"{ci_results['lower']*100:.1f}% - {ci_results['upper']*100:.1f}%",
                    delta=f"±{ci_results['margin']*100:.1f}%"
                )
        
        # Model comparison
        st.subheader("📊 All Approaches Comparison")
        comparison_df = pd.DataFrame({
            'Approach': ['Blending', 'ResNet', 'Stacking', 'Ensemble'],
            'Probability': [np.clip(p, 0, 1) for p in predictions.values()]
        })
        
        fig = px.bar(comparison_df, x='Approach', y='Probability', 
                     color='Probability', color_continuous_scale='RdYlGn_r',
                     title="Prediction Comparison Across All Approaches",
                     labels={'Probability': 'Flood Probability'})
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
        st.plotly_chart(fig, width='stretch')
        
        # Feature contribution analysis (simplified)
        st.subheader("🎯 Top Features Impact")
        base_sensitivity = {
            'Deforestation': 0.023405,
            'RiverManagement': 0.023315,
            'CoastalVulnerability': 0.023279,
            'WetlandLoss': 0.023251,
            'Encroachments': 0.023180
        }
        feature_impact_df = pd.DataFrame([
            {'Feature': feat, 'Impact Score': score} 
            for feat, score in sorted(base_sensitivity.items(), key=lambda x: x[1], reverse=True)[:5]
        ])
        fig_impact = px.bar(
            feature_impact_df, 
            x='Impact Score', 
            y='Feature', 
            orientation='h',
            title='Top 5 Most Influential Features',
            color='Impact Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_impact, use_container_width=True)

# ============================================================================
# ADVANCED ANALYSIS MODE
# ============================================================================
elif app_mode == "🔬 Advanced Analysis":
    st.header("Advanced Analysis & Visualization")
    
    base_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
        'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
        'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
    ]

    st.markdown("### 📂 Dataset for Analysis")
    uploaded_analysis_file = st.file_uploader(
        "Upload CSV for Advanced Analysis (uses your dataset)",
        type="csv",
        key="adv_analysis_upload"
    )

    analysis_feature_df = None
    if uploaded_analysis_file is not None:
        if st.session_state.get("adv_analysis_name") != uploaded_analysis_file.name:
            st.session_state["adv_analysis_df"] = pd.read_csv(uploaded_analysis_file)
            st.session_state["adv_analysis_name"] = uploaded_analysis_file.name

        uploaded_df = st.session_state["adv_analysis_df"]
        st.write(f"**Dataset loaded:** {len(uploaded_df)} rows × {len(uploaded_df.columns)} columns")

        missing_cols = set(base_features) - set(uploaded_df.columns)
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            max_rows = st.number_input(
                "Max rows to use for analysis (sampling for speed)",
                min_value=1,
                max_value=min(50000, len(uploaded_df)),
                value=min(2000, len(uploaded_df)),
                step=1,
                key="adv_analysis_max_rows",
                help="Enter any value between 1 and 50000 (e.g., 667, 42058, 189)"
            )

            analysis_feature_df = uploaded_df[base_features].copy()
            for col in analysis_feature_df.columns:
                median_val = analysis_feature_df[col].median()
                if pd.isna(median_val):
                    analysis_feature_df[col] = analysis_feature_df[col].fillna(0.5)
                else:
                    analysis_feature_df[col] = analysis_feature_df[col].fillna(median_val)

            # Sample exact number of rows specified by user
            actual_rows = min(int(max_rows), len(analysis_feature_df))
            if actual_rows < len(analysis_feature_df):
                analysis_feature_df = analysis_feature_df.sample(n=actual_rows, random_state=42)
                st.info(f"📊 Using {actual_rows} randomly sampled rows for analysis")
            else:
                st.info(f"📊 Using all {actual_rows} rows from dataset")
    
    tab1, tab2, tab3 = st.tabs([
        "📈 Sensitivity Analysis",
        "🎨 Feature Distribution",
        "🔗 Feature Correlation"
    ])
    
    with tab1:
        st.subheader("Sensitivity Analysis")
        st.write("Analyze how each feature affects flood probability using your uploaded dataset.")
        
        if analysis_feature_df is None:
            st.info("Upload a dataset above to run sensitivity analysis on your data.")
        else:
            sensitivities = {}
            input_df_engineered = advanced_features(analysis_feature_df)

            exclude = ['id', 'FloodProbability']
            feature_cols = [c for c in input_df_engineered.columns if c not in exclude]

            baseline_array = input_df_engineered[feature_cols].values
            scaler = models_dict['scaler']
            baseline_scaled = scaler.transform(baseline_array)
            baseline_pred = predict_with_approach(approach_num, baseline_scaled, models_dict)

            for feature in base_features:
                perturbed_df = analysis_feature_df.copy()
                perturbed_df[feature] = np.clip(perturbed_df[feature] + 0.1, 0.0, 1.0)

                perturbed_df_eng = advanced_features(perturbed_df)
                perturbed_array = perturbed_df_eng[feature_cols].values
                perturbed_scaled = scaler.transform(perturbed_array)
                perturbed_pred = predict_with_approach(approach_num, perturbed_scaled, models_dict)

                sensitivities[feature] = float(np.mean(np.abs(perturbed_pred - baseline_pred)))

            sens_df = pd.DataFrame({
                'Feature': list(sensitivities.keys()),
                'Impact': list(sensitivities.values())
            }).sort_values('Impact', ascending=False).head(10)

            fig = px.bar(sens_df, x='Feature', y='Impact',
                         color='Impact', color_continuous_scale='Reds',
                         title="Top 10 Features by Sensitivity (Dataset-Based)")
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Feature Value Distribution")
        if analysis_feature_df is None:
            st.info("Upload a dataset above to view feature distributions.")
        else:
            feature_to_plot = st.selectbox("Select feature to visualize:", base_features)

            feature_values = analysis_feature_df[feature_to_plot].dropna()
            if feature_values.empty:
                st.warning("No valid values for this feature in the dataset.")
            elif len(feature_values) < 2:
                st.warning(f"Only {len(feature_values)} data point available. Need at least 2 points for density curve.")
                # Show simple scatter plot for single point
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[feature_values.iloc[0]],
                    y=[1],
                    mode='markers',
                    marker=dict(size=15, color='#1565C0'),
                    name='Data Point'
                ))
                fig.update_layout(
                    title=f"Distribution of {feature_to_plot} (Dataset-Based)",
                    xaxis_title='Feature Value',
                    yaxis_title='Count',
                    showlegend=True
                )
                st.plotly_chart(fig, width='stretch')
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=feature_values,
                    nbinsx=50,
                    name='Frequency',
                    marker_color='#1565C0',
                    opacity=0.7,
                    yaxis='y'
                ))

                from scipy.stats import gaussian_kde
                kde = gaussian_kde(feature_values)
                x_range = np.linspace(feature_values.min(), feature_values.max(), 200)
                kde_values = kde(x_range)
                kde_scaled = kde_values * len(feature_values) * (1.0 / 50)

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_scaled,
                    mode='lines',
                    name='Density Curve',
                    line=dict(color='#FF6B00', width=3),
                    yaxis='y'
                ))

                fig.update_layout(
                    title=f"Distribution of {feature_to_plot} (Dataset-Based)",
                    xaxis_title='Feature Value',
                    yaxis_title='Frequency',
                    showlegend=True,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Feature Correlation Analysis")
        if analysis_feature_df is None:
            st.info("Upload a dataset above to view feature correlations.")
        else:
            corr_matrix = analysis_feature_df.corr()
        
        # Enhanced color scale for better visibility
            fig = px.imshow(corr_matrix,
                           labels=dict(x="Feature", y="Feature", color="Correlation"),
                           title="Feature Correlation Heatmap (Dataset-Based)",
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1,
                           aspect='auto')
        
        # Enhance colorbar and annotations
            fig.update_coloraxes(colorbar=dict(
                title="Correlation",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=400,
                tickmode='linear',
                tick0=-1,
                dtick=0.2
            ))
        
        # Add text annotations for better readability
            fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                             texttemplate='%{text}',
                             textfont_size=8)

            fig.update_layout(height=800)
            st.plotly_chart(fig, width='stretch')

# ============================================================================
# BATCH PROCESSING MODE
# ============================================================================
elif app_mode == "📊 Batch Processing":
    st.header("Batch Prediction")
    
    base_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
        'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
        'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
    ]
    
    st.markdown("### Upload CSV File")
    st.info("CSV should contain the 20 required feature columns (in any order). Missing values are auto-imputed. Values are auto-scaled.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        st.write(f"**File Info:** {len(batch_df)} rows × {len(batch_df.columns)} columns")
        st.dataframe(batch_df, width='stretch', height=500)
        
        # Check for FloodProbability column (case-insensitive)
        flood_prob_col = None
        for col in batch_df.columns:
            if col.lower() == 'floodprobability':
                flood_prob_col = col
                break
        
        if set(base_features).issubset(set(batch_df.columns)):
            # Handle missing values
            batch_df_clean = batch_df[base_features].copy()
            
            # Validate data types - convert to numeric and catch errors
            try:
                for col in base_features:
                    batch_df_clean[col] = pd.to_numeric(batch_df_clean[col], errors='coerce')
            except Exception as e:
                st.error(f"❌ Error converting features to numeric: {str(e)}")
                st.stop()
            
            missing_counts = batch_df_clean.isnull().sum()
            
            # Check for rows with all NaN
            all_nan_rows = batch_df_clean.isnull().all(axis=1).sum()
            if all_nan_rows > 0:
                st.warning(f"⚠️ Found {all_nan_rows} row(s) with all NaN values. These will be imputed with 0.5.")
            
            if missing_counts.sum() > 0:
                st.warning(f"⚠️ Found {missing_counts.sum()} missing values. Auto-imputing with median...")
                for col in base_features:
                    if batch_df_clean[col].isnull().sum() > 0:
                        median_val = batch_df_clean[col].median()
                        if pd.isna(median_val):  # All values in column are NaN
                            batch_df_clean[col].fillna(0.5, inplace=True)
                        else:
                            batch_df_clean[col].fillna(median_val, inplace=True)
            
            # Feature engineering
            batch_df_eng = advanced_features(batch_df_clean)
            exclude = ['id', 'FloodProbability']
            feature_cols = [c for c in batch_df_eng.columns if c not in exclude]
            
            # Predictions
            batch_array = batch_df_eng[feature_cols].values
            scaler = models_dict['scaler']
            batch_scaled = scaler.transform(batch_array)
            
            batch_predictions = predict_with_approach(approach_num, batch_scaled, models_dict)
            batch_predictions = np.clip(batch_predictions, 0, 1)
            
            # Results
            results_df = batch_df.copy()
            results_df['Flood_Probability_Predicted'] = batch_predictions
            results_df['Risk_Level'] = results_df['Flood_Probability_Predicted'].apply(risk_label)
            
            st.success(f"✅ Predictions generated for {len(results_df)} samples!")
            st.dataframe(results_df, width='stretch')
            
            # Statistics
            st.subheader("Summary Statistics")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Avg Probability", f"{batch_predictions.mean():.3f}")
            with cols[1]:
                st.metric("Max Probability", f"{batch_predictions.max():.3f}")
            with cols[2]:
                st.metric("Min Probability", f"{batch_predictions.min():.3f}")
            with cols[3]:
                high_risk = sum(batch_predictions >= 0.7)
                st.metric("High Risk Count", high_risk)
            
            # Model accuracy and RMSE if FloodProbability column exists
            if flood_prob_col is not None and flood_prob_col in batch_df.columns:
                st.subheader("📊 Model Performance Metrics")
                true_values = batch_df[flood_prob_col].values
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(true_values, batch_predictions))
                mae = mean_absolute_error(true_values, batch_predictions)
                
                # Binary accuracy (threshold at 0.5)
                binary_true = (true_values >= 0.5).astype(int)
                binary_pred = (batch_predictions >= 0.5).astype(int)
                accuracy = accuracy_score(binary_true, binary_pred)
                
                # R-squared
                from sklearn.metrics import r2_score
                r2 = r2_score(true_values, batch_predictions)
                
                perf_cols = st.columns(4)
                with perf_cols[0]:
                    st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
                with perf_cols[1]:
                    st.metric("📉 RMSE", f"{rmse:.4f}")
                with perf_cols[2]:
                    st.metric("📊 MAE", f"{mae:.4f}")
                with perf_cols[3]:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Visualization: Actual vs Predicted
                comparison_data = pd.DataFrame({
                    'Actual': true_values,
                    'Predicted': batch_predictions
                })
                
                fig = px.scatter(comparison_data, x='Actual', y='Predicted',
                                title=f"Actual vs Predicted Flood Probability ({len(comparison_data)} samples)",
                                labels={'Actual': 'True Value', 'Predicted': 'Prediction'})
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                             line=dict(color="red", width=2, dash="dash"))
                st.plotly_chart(fig, width='stretch')
            
            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name='flood_predictions.csv',
                mime='text/csv',
                width='stretch'
            )
        else:
            missing = set(base_features) - set(batch_df.columns)
            st.error(f"❌ Missing columns: {missing}")

# ============================================================================
# HELP & INFO MODE
# ============================================================================
else:  # Help & Info
    st.header("📚 Help & Information")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 About Normalization",
        "📖 Features Guide",
        "🎯 Model Info",
        "⚙️ Troubleshooting",
        "📊 Dataset & Project"
    ])
    
    with tab1:
        st.subheader("Why Normalize Inputs to [0, 1]?")
        st.markdown("""
        **The Requirement:**
        - All inputs must be between 0 and 1
        
        **Why?**
        1. **Training Assumption**: Models trained on normalized data [0, 1]
        2. **Feature Engineering**: Engineered features computed from normalized base features
        3. **StandardScaler Calibration**: Fitted on normalized data; raw values break statistics
        4. **Model Accuracy**: Unnormalized inputs → incorrect predictions
        
        **Example Normalization:**
        - Temperature (0-50°C) → Divide by 50 → [0, 1]
        - Rainfall (0-500mm) → Divide by 500 → [0, 1]
        - Population (0-1M) → Divide by 1M → [0, 1]
        """)
    
    with tab2:
        st.subheader("📚 Detailed Feature Dictionary")
        
        if ENHANCEMENTS_AVAILABLE and FEATURE_DICTIONARY:
            # Organized feature dictionary
            st.markdown("""
            ### 20 Base Input Features (All normalized to [0, 1])
            Each feature represents a critical factor influencing flood probability. 
            Higher values indicate greater risk contribution for most features.
            """)
            
            # Search/filter features
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("🔍 Search features:", placeholder="e.g., 'Deforestation'").lower()
            
            # Display features with filtering
            for feature_name, feature_info in FEATURE_DICTIONARY.items():
                if search_term and search_term not in feature_name.lower() and search_term not in feature_info.get('description', '').lower():
                    continue
                    
                with st.expander(f"🎯 **{feature_name}**", expanded=False):
                    st.markdown(f"""
                    **Description:** {feature_info['description']}
                    
                    **Impact on Floods:** {feature_info['impact']}
                    
                    **Data Type:** {feature_info['unit']} | **Range:** {feature_info['range']}
                    """)
            
            st.markdown("---")
            st.subheader("📊 Engineered Features (67 Total)")
            st.markdown("""
            Original 20 features are transformed into 67 engineered features to capture complex patterns:
            
            - **20 Base Features** + **47 Engineered Features** = **67 Total**
            """)
            
            if ENGINEERED_FEATURES_GROUPS:
                total_eng = sum(len(v) for v in ENGINEERED_FEATURES_GROUPS.values())
                st.markdown(f"**Total Engineered Features: {total_eng}**")
                
                for group_name, feature_list in ENGINEERED_FEATURES_GROUPS.items():
                    features_display = ', '.join(feature_list[:6]) + ('...' if len(feature_list) > 6 else '')
                    st.write(f"**{group_name}** ({len(feature_list)}): {features_display}")
        else:
            # Fallback to simple descriptions
            features_info = {
                'MonsoonIntensity': 'Intensity of monsoon rainfall',
                'TopographyDrainage': 'Terrain drainage efficiency',
                'RiverManagement': 'River management quality',
                'Deforestation': 'Level of deforestation',
                'Urbanization': 'Urban development level',
                'ClimateChange': 'Impact of climate change',
                'DamsQuality': 'Dam infrastructure quality',
                'Siltation': 'River siltation level',
                'AgriculturalPractices': 'Sustainability of agricultural practices',
                'Encroachments': 'Encroachment on waterways',
                'IneffectiveDisasterPreparedness': 'Disaster preparedness level',
                'DrainageSystems': 'Drainage system efficiency',
                'CoastalVulnerability': 'Coastal area vulnerability',
                'Landslides': 'Landslide risk',
                'Watersheds': 'Watershed condition',
                'DeterioratingInfrastructure': 'Infrastructure deterioration',
                'PopulationScore': 'Population in flood-prone areas',
                'WetlandLoss': 'Wetland ecosystem loss',
                'InadequatePlanning': 'Urban planning adequacy',
                'PoliticalFactors': 'Political/governance factors'
            }
            
            for feature, description in features_info.items():
                st.write(f"**{feature}**: {description}")
    
    with tab3:
        st.subheader("Model Approaches")
        st.markdown("""
        **Approach 1: Blending + Ridge**
        - Uses: LightGBM, CatBoost, XGBoost, HistGradientBoosting
        - Fast and stable
        
        **Approach 2: ResNet Neural Network**
        - Deep learning with residual connections
        - Captures complex feature interactions
        
        **Approach 3: Stacking (Hybrid)**
        - Combines ResNet + CatBoost + XGBoost
        - Strong hybrid performance
        
        **Approach 4: Final Ensemble (Recommended)**
        - 70% Approach 3 + 30% Approach 2
        - Best overall accuracy and robustness
        """)
    
    with tab4:
        st.subheader("Troubleshooting")
        st.markdown("""
        **"Inputs must be normalized between 0 and 1"**
        - Ensure all input values are between 0 and 1
        - Normalize: value / max_value
        
        **"Missing values"**
        - Leave fields blank or use NaN
        - Auto-imputation with median values
        
        **"Models not loading"**
        - Run: `python flood_prediction_model.py`
        - Ensure model files exist in project folder
        
        **"Slow predictions"**
        - First run loads models (one-time)
        - Subsequent predictions are faster
        """)
        
    with tab5:
        st.subheader("Training Dataset Overview")

        base_features = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
            'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
            'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
            'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
            'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
        ]

        train_path = "train.csv"
        if not os.path.exists(train_path):
            st.info("📊 **Training Data Overview**")
            st.markdown("""
            The training dataset (`train.csv`) is not included in this deployment to reduce repository size.
            
            **Pre-computed analytics are available below**, computed on the full training dataset (~1.4M rows).
            
            To access the raw training data:
            - Download from [Kaggle - Flood Prediction Train and Test Dataset](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset)
            - Place `train.csv` in the project root directory
            - Restart the app
            """)
            
            st.markdown("---")
            st.subheader("📈 Pre-computed Feature-Target Correlations")
            
            # Show pre-computed correlation if available
            corr_file = "train_correlation_matrix.pkl"
            if os.path.exists(corr_file):
                try:
                    corr_matrix = joblib.load(corr_file)
                    if ENHANCEMENTS_AVAILABLE:
                        corr_df = compute_feature_target_correlation(corr_matrix)
                        st.dataframe(corr_df.head(20), use_container_width=True)
                        
                        fig = px.bar(corr_df.head(10), x='Feature', y='Correlation',
                                   color='Correlation', color_continuous_scale='RdBu_r',
                                   title="Top 10 Features by Correlation with FloodProbability")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Enhanced analysis unavailable")
                except Exception as e:
                    st.warning(f"Could not load pre-computed correlation: {e}")
            else:
                st.warning("Pre-computed correlation file not found.")
            
            st.markdown("---")
            st.subheader("🎯 Pre-computed Feature Sensitivity")
            
            # Show pre-computed sensitivity if available
            sens_file = "train_sensitivity_approach4.pkl"
            if os.path.exists(sens_file):
                try:
                    sensitivities = joblib.load(sens_file)
                    sens_plot_df = pd.DataFrame({
                        'Feature': list(sensitivities.keys()),
                        'Impact': list(sensitivities.values())
                    }).sort_values('Impact', ascending=False).head(10)
                    
                    fig = px.bar(sens_plot_df, x='Feature', y='Impact',
                               color='Impact', color_continuous_scale='Reds',
                               title="Top 10 Features by Sensitivity (Computed on 1.4M rows)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(sens_plot_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not load pre-computed sensitivity: {e}")
            else:
                st.warning("Pre-computed sensitivity file not found.")
            
            st.markdown("---")
            st.markdown("""
            **Note:** All pre-computed analytics were generated from the complete training dataset.
            For interactive analysis with your own data, use the **Advanced Analysis** mode.
            """)
            
        else:
            # Initialize session state for train data
            if 'train_dataset_loaded' not in st.session_state:
                st.session_state['train_dataset_loaded'] = False
            
            # Auto-load data on first visit
            if not st.session_state['train_dataset_loaded']:
                with st.spinner("Loading train.csv..."):
                    usecols = base_features + ["FloodProbability"]
                    st.session_state['train_df_cached'] = load_train_data(train_path, usecols=usecols)
                    st.session_state['train_dataset_loaded'] = True

            # Display data
            train_df = st.session_state['train_df_cached']
            
            st.write(f"**Rows:** {len(train_df):,} | **Columns:** {len(train_df.columns)}")
            missing_pct = (train_df.isna().mean() * 100).round(2)
            missing_df = missing_pct.reset_index()
            missing_df.columns = ["Feature", "Missing (%)"]
            st.dataframe(missing_df, width='stretch', hide_index=True)

            st.markdown("### Feature Distribution (Train Data)")
            dist_feature = st.selectbox(
                "Select feature to visualize:",
                base_features,
                key="train_dist_feature"
            )

            dist_values = train_df[dist_feature].dropna()
            if dist_values.empty:
                st.warning("No valid values for this feature in train.csv.")
            elif len(dist_values) < 2:
                st.warning(f"Only {len(dist_values)} data point available. Need at least 2 points for density curve.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=dist_values,
                    nbinsx=50,
                    name='Frequency',
                    marker_color='#1565C0',
                    opacity=0.7,
                    yaxis='y'
                ))

                from scipy.stats import gaussian_kde
                kde = gaussian_kde(dist_values)
                x_range = np.linspace(dist_values.min(), dist_values.max(), 200)
                kde_values = kde(x_range)
                kde_scaled = kde_values * len(dist_values) * (1.0 / 50)

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_scaled,
                    mode='lines',
                    name='Density Curve',
                    line=dict(color='#FF6B00', width=3),
                    yaxis='y'
                ))

                fig.update_layout(
                    title=f"Distribution of {dist_feature} (Train Data)",
                    xaxis_title='Feature Value',
                    yaxis_title='Frequency',
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')

            st.markdown("### Feature Correlation (Train Data)")
            
            # Try loading pre-computed correlation matrix
            corr_matrix, is_precomputed = load_correlation_matrix()
            
            if corr_matrix is None:
                # Fallback: compute on-the-fly from train_df
                with st.spinner("Computing correlation matrix from all rows..."):
                    corr_matrix = train_df[base_features].corr()
                st.info("💡 Tip: Run `python precompute_correlation.py` to pre-compute for instant loading")
                title_suffix = f"{len(train_df):,} rows"
            else:
                title_suffix = "1.4M+ rows (pre-computed)"

            fig = px.imshow(corr_matrix,
                           labels=dict(x="Feature", y="Feature", color="Correlation"),
                           title=f"Feature Correlation Heatmap ({title_suffix})",
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1,
                           aspect='auto')

            fig.update_coloraxes(colorbar=dict(
                title="Correlation",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=400,
                tickmode='linear',
                tick0=-1,
                dtick=0.2
            ))

            fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                             texttemplate='%{text}',
                             textfont_size=8)

            fig.update_layout(height=800)
            st.plotly_chart(fig, width='stretch')

            st.markdown("### Feature Sensitivity (Train Data - All 1.4M Rows)")
            
            # Try to load pre-computed sensitivity
            sens_results, loaded = load_sensitivity_results()
            
            if loaded and sens_results is not None:
                # Display pre-computed results (instant)
                
                if isinstance(sens_results, dict):
                    # If loaded from PKL (dict format)
                    sens_plot_df = pd.DataFrame({
                        'Feature': list(sens_results.keys()),
                        'Impact': list(sens_results.values())
                    }).sort_values('Impact', ascending=False).head(10)
                else:
                    # If loaded from CSV (DataFrame format)
                    sens_plot_df = sens_results.sort_values('Impact', ascending=False).head(10)
                
                fig = px.bar(sens_plot_df, x='Impact', y='Feature', orientation='h',
                             color='Impact', color_continuous_scale='Reds',
                             title="Top 10 Features by Sensitivity (Approach 4, 1.4M rows)")
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.dataframe(sens_plot_df.reset_index(drop=True), use_container_width=True)
            else:
                st.warning("⚠️ Pre-computed sensitivity files not found. Computing on sample data...")
                
                # Fallback: compute on sample if files not available
                base_features = [c for c in train_df.columns 
                               if c not in ['id', 'FloodProbability']]
                
                sens_rows = st.number_input(
                    "Max rows for sensitivity analysis",
                    min_value=200,
                    max_value=min(5000, len(train_df)),
                    value=min(1000, len(train_df)),
                    step=100,
                    key="train_sens_rows"
                )

                sens_df = train_df[base_features].sample(
                    n=int(sens_rows),
                    random_state=42
                )

                for col in sens_df.columns:
                    median_val = sens_df[col].median()
                    if pd.isna(median_val):
                        sens_df[col] = sens_df[col].fillna(0.5)
                    else:
                        sens_df[col] = sens_df[col].fillna(median_val)

                sens_eng = advanced_features(sens_df)
                exclude = ['id', 'FloodProbability']
                feature_cols = [c for c in sens_eng.columns if c not in exclude]

                baseline_array = sens_eng[feature_cols].values
                scaler = models_dict['scaler']
                baseline_scaled = scaler.transform(baseline_array)
                baseline_pred = predict_with_approach(4, baseline_scaled, models_dict)

                sensitivities = {}
                for feature in base_features:
                    perturbed_df = sens_df.copy()
                    perturbed_df[feature] = np.clip(perturbed_df[feature] + 0.1, 0.0, 1.0)

                    perturbed_eng = advanced_features(perturbed_df)
                    perturbed_array = perturbed_eng[feature_cols].values
                    perturbed_scaled = scaler.transform(perturbed_array)
                    perturbed_pred = predict_with_approach(4, perturbed_scaled, models_dict)

                    sensitivities[feature] = float(np.mean(np.abs(perturbed_pred - baseline_pred)))

                sens_plot_df = pd.DataFrame({
                    'Feature': list(sensitivities.keys()),
                    'Impact': list(sensitivities.values())
                }).sort_values('Impact', ascending=False).head(10)

                fig = px.bar(sens_plot_df, x='Feature', y='Impact',
                             color='Impact', color_continuous_scale='Reds',
                             title="Top 10 Features by Sensitivity (Train Data)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Target Variable Distribution")
            
            # FloodProbability distribution
            flood_prob = train_df['FloodProbability'].dropna()
            mean_flood = flood_prob.mean()
            
            # Create histogram with KDE overlay
            fig_dist = go.Figure()
            
            # Histogram bars
            fig_dist.add_trace(go.Histogram(
                x=flood_prob,
                nbinsx=30,
                name='Frequency',
                marker_color='rgba(255, 200, 0, 0.7)',
                showlegend=True
            ))
            
            # KDE curve overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(flood_prob)
            x_range = np.linspace(flood_prob.min(), flood_prob.max(), 100)
            kde_values = kde(x_range)
            # Scale KDE to match histogram height
            kde_values = kde_values * len(flood_prob) * (flood_prob.max() - flood_prob.min()) / 30
            
            fig_dist.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name='KDE',
                line=dict(color='rgb(128, 0, 128)', width=3),
                showlegend=True
            ))
            
            # Add mean line
            fig_dist.add_vline(
                x=mean_flood,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean = {mean_flood:.3f}",
                annotation_position="top right"
            )
            
            fig_dist.update_layout(
                title="FloodProbability Distribution (Train Data)",
                xaxis_title="FloodProbability",
                yaxis_title="Density",
                hovermode='x unified',
                height=500,
                barmode='overlay'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown(f"""
            **Distribution Statistics:**
            - Mean: {mean_flood:.4f}
            - Std Dev: {flood_prob.std():.4f}
            - Min: {flood_prob.min():.4f}
            - Max: {flood_prob.max():.4f}
            - Median: {flood_prob.median():.4f}
            """)

            st.markdown("---")
            st.subheader("📊 Training Data Stratification")
            st.markdown("Distribution of target variable across different probability ranges:")
            
            if ENHANCEMENTS_AVAILABLE:
                # Analyze class distribution
                stratification_df = analyze_class_distribution(train_df, target_col='FloodProbability', n_bins=10)
                if stratification_df is not None:
                    st.dataframe(stratification_df, use_container_width=True)
                    
                    # Visualization
                    fig_strat = px.bar(
                        stratification_df,
                        x='Probability Range',
                        y='Percentage',
                        title='Target Variable Distribution by Probability Range',
                        labels={'Percentage': 'Percentage of Samples (%)'},
                        color='Percentage',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
            else:
                st.info("Enhanced analysis unavailable")

            st.markdown("---")
            st.subheader("🔗 Feature-Target Correlation Analysis")
            st.markdown("Correlation between each base feature and FloodProbability:")
            
            if ENHANCEMENTS_AVAILABLE:
                # Compute feature-target correlations
                base_features = [
                    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
                    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
                    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
                    'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
                    'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
                ]
                
                corr_df = compute_feature_target_correlation(train_df, base_features, target_col='FloodProbability')
                
                # Display as table
                st.dataframe(corr_df.head(10), use_container_width=True)
                
                # Visualization
                fig_corr = px.bar(
                    corr_df.head(15),
                    x='Correlation',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Features by Correlation with FloodProbability',
                    color='Correlation',
                    color_continuous_scale='RdBu',
                    labels={'Correlation': 'Pearson Correlation Coefficient'}
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Enhanced analysis unavailable")

        st.markdown("---")
        st.subheader("Project Pipeline Summary")
        st.markdown("""
        **1. Data Preprocessing**
        - Remove duplicates and handle missing values
        - Normalize all input features into the [0, 1] range

        **2. Feature Engineering**
        - Expand 20 base features into 67 engineered features
        - Add statistical, dispersion, norm, entropy, and custom coefficients

        **3. Modeling Approaches**
        - **Approach 1:** Blending of gradient boosting models + RidgeCV meta-learner
        - **Approach 2:** ResNet-style neural network for tabular data
        - **Approach 3:** Stacking (ResNet + CatBoost + XGBoost) + RidgeCV
        - **Approach 4:** Final ensemble (70% Approach 3 + 30% Approach 2)

        **4. Deployment**
        - Streamlit web app with 4 interactive modes
        - Batch processing for large datasets (1.4M+ rows tested)
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>🌊 <b>RiverGuard</b> - Flood Intelligence System | v2.0 Enhanced</p>
        <p style="font-size: 0.9rem;">Powered by RIVERGUARD</p>
    </div>
""", unsafe_allow_html=True)
