import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_generator import LOCATIONS

condition_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}


def preprocess(df, test_size=0.2, seed=42):
    df = df.copy()

    df[['Location', 'Condition']] = SimpleImputer(strategy='most_frequent').fit_transform(
        df[['Location', 'Condition']]
    )
    df['Condition'] = df['Condition'].map(condition_map)

    numeric_cols = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Has_pool', 'Garage_spaces', 'Age']
    df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])

    df['Rooms_total'] = df['Bedrooms'] + df['Bathrooms']
    df['Is_new'] = (df['Age'] <= 5).astype(int)
    df['Luxury_score'] = (
        df['Area_sqft'] / 2000
        + df['Bedrooms'] / 5
        + df['Garage_spaces'] / 3
        + df['Has_pool']
    ) / 4
    df['Area_per_bedroom'] = df['Area_sqft'] / (df['Bedrooms'] + 1)
    df['Log_area'] = np.log1p(df['Area_sqft'])

    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    feature_columns = [col for col in df.columns if col != 'Price']
    X = df[feature_columns]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_columns,
        scaler,
        LOCATIONS,
        condition_map,
    )


def transform_new_house(house, feature_columns, location_categories, condition_map):
    df = pd.DataFrame([house])
    df['Location'] = pd.Categorical(df['Location'], categories=location_categories)
    df['Condition'] = df['Condition'].map(condition_map)
    df['Has_pool'] = df['Has_pool'].astype(int)

    df['Rooms_total'] = df['Bedrooms'] + df['Bathrooms']
    df['Is_new'] = (df['Age'] <= 5).astype(int)
    df['Luxury_score'] = (
        df['Area_sqft'] / 2000
        + df['Bedrooms'] / 5
        + df['Garage_spaces'] / 3
        + df['Has_pool']
    ) / 4
    df['Area_per_bedroom'] = df['Area_sqft'] / (df['Bedrooms'] + 1)
    df['Log_area'] = np.log1p(df['Area_sqft'])

    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]
