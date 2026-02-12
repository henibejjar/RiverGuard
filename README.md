# RiverGuard Flood Intelligence

Flood probability prediction system with a Streamlit web app and four ML approaches. Runs single predictions, advanced analysis, and large batch scoring.

🌐 **[Live Demo](https://heni-bejjar-riverguard.streamlit.app/)** | 📊 **[Dataset on Kaggle](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset)**

## Quick Start

### Option 1: Use Pre-trained Models (Fastest)

```bash
pip install -r requirements.txt
streamlit run app_flood.py
```

Open `http://localhost:8501`.

**All trained models are included** in the repository (~14 MB). No training required!

### Option 2: Train Models Yourself

Download `train.csv` and `test.csv` from: [Kaggle - Flood Prediction Train and Test Dataset](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset)

Place both files in the project root directory, then:

```bash
pip install -r requirements.txt
python flood_prediction_model.py  # Takes 5-10 minutes
streamlit run app_flood.py
```

## Data Requirements

`train.csv` and `test.csv` must include the 20 base features:

```
MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation,
Urbanization, ClimateChange, DamsQuality, Siltation, AgriculturalPractices,
Encroachments, IneffectiveDisasterPreparedness, DrainageSystems,
CoastalVulnerability, Landslides, Watersheds, DeterioratingInfrastructure,
PopulationScore, WetlandLoss, InadequatePlanning, PoliticalFactors
```

`train.csv` must also include `FloodProbability` (0-1). `test.csv` must include `id`.

## App Features

### Navigation Modes
- **Quick Prediction:** Single input prediction with 95% confidence intervals and feature impact analysis
- **Advanced Analysis:** Dataset-based sensitivity, distribution, correlation analysis
- **Batch Processing:** CSV upload (200MB), performance metrics if target present, tested on 1.4M+ rows
- **Help & Info:** Feature dictionary, model explanations, and training dataset overview

### Real-Time Analytics
- **95% Confidence Intervals:** Uncertainty bounds calculated from all 4 model approaches
- **Feature Impact Visualization:** Top 5 most influential features (pre-computed on 1.4M rows)
- **Pre-computed Analytics:** Instant correlation matrix, sensitivity scores, target distribution

## Modeling Summary

- **Approach 1:** Blending + Ridge meta-learner (LightGBM, CatBoost, XGBoost, HistGradientBoosting)
- **Approach 2:** ResNet tabular network with residual connections
- **Approach 3:** Stacking (ResNet + CatBoost + XGBoost) + Ridge meta-learner
- **Approach 4:** Ensemble (70% Approach 3 + 30% Approach 2) — **Recommended default**

## Data & Model Files

### Dataset
Download training and test data from [Kaggle](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset):
- `train.csv` (~1.4M rows, 21 columns)
- `test.csv` (20 base features + id column)

### Included Model Files
This repository includes all pre-trained models (~14 MB):
- `approach1_models.pkl` - Blending ensemble models (4.05 MB)
- `approach2_resnet.pt` - ResNet neural network weights (2.13 MB)
- `approach3_models.pkl` - Stacking ensemble models (2.58 MB)
- `flood_scaler.pkl` - Feature scaler (min-max normalization)
- `flood_model_metadata.pkl` - Training metadata
- `all_predictions.pkl` - Model predictions archive (4.95 MB)
- `train_correlation_matrix.pkl` - Pre-computed correlation matrix
- `train_sensitivity_approach4.pkl` - Pre-computed sensitivity scores

**Ready to use immediately** - no training required! Dataset files only needed if you want to retrain models.

## License

This project is licensed under the **MIT License** - free to use, modify, and distribute with attribution.

Dataset available on [Kaggle](https://www.kaggle.com/datasets/henibejar/flood-prediction-train-and-test-dataset) under MIT License.
