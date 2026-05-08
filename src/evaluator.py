import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def get_top_features(model, feature_columns, top_n=12):
    if hasattr(model, 'feature_importances_'):
        scores = model.feature_importances_
    elif hasattr(model, 'coef_'):
        scores = np.abs(model.coef_)
    else:
        return []

    feature_scores = sorted(
        zip(feature_columns, scores), key=lambda item: item[1], reverse=True
    )
    return feature_scores[:top_n]
