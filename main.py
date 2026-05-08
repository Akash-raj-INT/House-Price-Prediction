from src.data_generator import generate_dataset
from src.preprocessor import preprocess, transform_new_house
from src.trainer import train_models, tune_xgboost
from src.evaluator import evaluate_model, get_top_features
from src.visualizer import plot_eda, plot_model_performance


def main():
    df = generate_dataset(n=1000, seed=42)

    plot_eda(df)

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_columns,
        scaler,
        location_categories,
        condition_map,
    ) = preprocess(df, test_size=0.2, seed=42)

    models, cv_results = train_models(
        X_train,
        X_train_scaled,
        y_train,
        feature_columns,
        seed=42,
    )

    best_name = max(cv_results, key=lambda name: cv_results[name]['cv_mean'])
    best_model = models[best_name]

    tuned = tune_xgboost(X_train, y_train, seed=42)
    if tuned is not None:
        models['XGBoost Tuned'] = tuned

    test_results = {}
    for name, model in models.items():
        test_results[name] = evaluate_model(
            model,
            X_test_scaled if name in ['Linear Regression', 'Ridge', 'Lasso'] else X_test,
            y_test,
        )

    print('\n=== Test Set Performance ===')
    for name, metrics in test_results.items():
        print(f"{name}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    best_test_name = max(test_results, key=lambda name: test_results[name]['r2'])
    best_test_model = models[best_test_name]
    print(f"\nBest test model: {best_test_name}")

    top_features = get_top_features(best_test_model, feature_columns)
    print('\nTop features for best model:')
    for feature, score in top_features[:5]:
        print(f"  {feature}: {score:.4f}")

    y_pred = best_test_model.predict(
        X_test_scaled if best_test_name in ['Linear Regression', 'Ridge', 'Lasso'] else X_test
    )
    plot_model_performance(test_results, y_test, y_pred, best_test_model, feature_columns)

    sample_house = {
        'Area_sqft': 2800,
        'Bedrooms': 4,
        'Bathrooms': 3,
        'Location': 'Suburbs',
        'Condition': 'Excellent',
        'Has_pool': 1,
        'Garage_spaces': 2,
        'Age': 4,
    }

    X_custom = transform_new_house(
        sample_house,
        feature_columns,
        location_categories,
        condition_map,
    )
    if best_test_name in ['Linear Regression', 'Ridge', 'Lasso']:
        X_custom = scaler.transform(X_custom)

    prediction = best_test_model.predict(X_custom)[0]
    print(f"\nLive prediction demo: estimated house price = ${prediction:,.2f}")


if __name__ == '__main__':
    main()
