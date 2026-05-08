# 🏠 House Price Prediction — ML Project

A complete machine learning pipeline that generates a synthetic real estate dataset, performs EDA, preprocesses and trains models, tunes XGBoost, visualizes performance, and runs a live prediction demo.

## 📁 Project Structure

```
HousePricePrediction/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── visualizer.py
│   └── predictor.py
└── outputs/  # created automatically on run
```

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python main.py
```

## What is included

- Synthetic 1,000-sample dataset with realistic features and ~5% missing values
- EDA plots saved to `outputs/eda_plots.png`
- Preprocessing with median imputation and feature engineering
- Train/test split and scaling for linear models
- Comparison of 6 models with cross-validation
- XGBoost hyperparameter tuning
- Performance visualization saved to `outputs/model_performance.png`
- Live prediction demo for a custom house
