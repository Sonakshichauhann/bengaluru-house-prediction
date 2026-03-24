
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# Data Cleaning + Feature Engineering
# -------------------------------
def clean_and_engineer_data(df):
    df = df.dropna()

    # Remove unrealistic BHK values
    df = df[df['BHK'] <= 10]

    # Remove very small sqft entries
    df = df[df['total_sqft'] > 300]

    # Price per sqft
    df['price_per_sqft'] = (df['Price'] * 100000) / df['total_sqft']

    # Outlier removal on Price
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 1.5*IQR) & (df['Price'] <= Q3 + 1.5*IQR)]

    # Remove price_per_sqft outliers within each location
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    df = df_out

    # Sanity check: sqft per BHK
    df = df[df['total_sqft'] / df['BHK'] >= 300]

    # Remove BHK outliers
    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('BHK'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df['price_per_sqft']),
                    'std': np.std(bhk_df['price_per_sqft']),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('BHK'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values)
        return df.drop(exclude_indices, axis='index')

    df = remove_bhk_outliers(df)

    # Feature: bath_per_bhk
    df['bath_per_bhk'] = df['bath'] / df['BHK']

    # Encode location (reduce rare ones)
    location_stats = df['location'].value_counts(ascending=False)
    locations_less_than_10 = location_stats[location_stats <= 10].index
    df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

    return df

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Rename target column to match pipeline
df.rename(columns={'price': 'Price'}, inplace=True)

# Drop unused columns
df.drop(columns=['area_type','availability','society','balcony'], inplace=True)

# Fill missing values
df['location'] = df['location'].fillna('Sarjapur Road')
df['size'] = df['size'].fillna('2 BHK')

bath_median = df['bath'].median()
df['bath'] = df['bath'].fillna(bath_median)

# Extract BHK
df['BHK'] = df['size'].str.split().str.get(0).astype(int)
df.drop(columns=['size'],inplace=True)

# Convert sqft to numeric
def convertRange(x):
    try:
        temp = x.split('-')
        if len(temp) == 2:
            return (float(temp[0]) + float(temp[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convertRange)
df = df.dropna(subset=['total_sqft'])

# Clean + Engineer
df = clean_and_engineer_data(df)

# -------------------------------
# Model Training
# -------------------------------
X = df.drop(['Price'], axis=1)
X = pd.get_dummies(X, columns=['location'], drop_first=True)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)



print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))

cv_scores = cross_val_score(rf, X, y, cv=5)
print("Random Forest Cross-val Accuracy:", np.mean(cv_scores))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_price(location, sqft, bhk, bath, model=rf):
    # Create input row
    x = pd.DataFrame([[sqft, bhk, bath, bath/bhk, location]],
                     columns=['total_sqft','BHK','bath','bath_per_bhk','location'])
    
    # Encode location
    x = pd.get_dummies(x, columns=['location'], drop_first=True)

    # Align with training columns
    x = x.reindex(columns=X.columns, fill_value=0)

    # Predict
    return model.predict(x)[0]

# Example usage:
print("\nExample Prediction:")
print("Price prediction (Whitefield, 1200 sqft, 2 BHK, 2 bath) →",
      predict_price("Whitefield", 1200, 2, 2), "Lakhs")

import joblib
joblib.dump(rf, "model.pkl")
# Save training columns
joblib.dump(list(X.columns), "model_columns.pkl")

# Save unique locations
joblib.dump(list(df['location'].unique()), "model_locations.pkl")







