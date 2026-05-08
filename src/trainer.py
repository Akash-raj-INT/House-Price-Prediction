from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb


def train_models(X_train, X_train_scaled, y_train, feature_columns, seed=42):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=seed),
        'Lasso': Lasso(alpha=0.002, random_state=seed, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=seed),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=seed),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=150,
            random_state=seed,
            objective='reg:squarederror',
            eval_metric='rmse',
        ),
    }

    cv_results = {}
    for name, model in models.items():
        X_input = X_train_scaled if name in ['Linear Regression', 'Ridge', 'Lasso'] else X_train
        scores = cross_val_score(model, X_input, y_train, cv=5, scoring='r2', n_jobs=-1)
        model.fit(X_input, y_train)
        cv_results[name] = {
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
        }

    return models, cv_results


def tune_xgboost(X_train, y_train, seed=42):
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
    }

    grid = GridSearchCV(
        xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=seed,
            eval_metric='rmse',
        ),
        param_grid,
        scoring='r2',
        cv=4,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)
    print(f"\nBest XGBoost params: {grid.best_params_}, R2={grid.best_score_:.4f}")
    return grid.best_estimator_
