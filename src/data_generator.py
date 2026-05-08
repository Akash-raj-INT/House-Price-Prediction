import numpy as np
import pandas as pd

LOCATIONS = ['Downtown', 'Suburbs', 'Rural', 'Waterfront', 'Uptown']
CONDITIONS = ['Poor', 'Fair', 'Good', 'Excellent']


def generate_dataset(n=1000, seed=42):
    np.random.seed(seed)

    area = np.random.normal(loc=2200, scale=650, size=n).clip(500, 5500)
    bedrooms = np.random.choice([2, 3, 4, 5], size=n, p=[0.2, 0.4, 0.3, 0.1])
    bathrooms = np.random.choice([1, 2, 3], size=n, p=[0.25, 0.55, 0.20])
    location = np.random.choice(LOCATIONS, size=n, p=[0.2, 0.35, 0.2, 0.1, 0.15])
    condition = np.random.choice(CONDITIONS, size=n, p=[0.1, 0.25, 0.45, 0.2])
    has_pool = np.random.binomial(1, 0.18, size=n)
    garage_spaces = np.random.choice([0, 1, 2, 3], size=n, p=[0.1, 0.35, 0.4, 0.15])
    age = np.random.randint(0, 80, size=n)

    price_base = 45000 + area * 120
    location_multiplier = np.array([1.5 if loc == 'Downtown' else 1.25 if loc == 'Waterfront' else 1.1 if loc == 'Uptown' else 1.0 if loc == 'Suburbs' else 0.85 for loc in location])
    condition_multiplier = np.array([0.8 if cond == 'Poor' else 0.95 if cond == 'Fair' else 1.1 if cond == 'Good' else 1.35 for cond in condition])

    price = (
        price_base
        * location_multiplier
        * condition_multiplier
        * (1 + 0.02 * has_pool)
        * (1 + 0.03 * garage_spaces)
        * np.where(age <= 5, 1.05, 0.98)
        + np.random.normal(0, 20000, size=n)
    ).clip(20000, 1500000)

    df = pd.DataFrame(
        {
            'Area_sqft': np.round(area, 0),
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Location': location,
            'Condition': condition,
            'Has_pool': has_pool,
            'Garage_spaces': garage_spaces,
            'Age': age,
            'Price': np.round(price, 0),
        }
    )

    for col in ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Location', 'Condition']:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan

    return df
