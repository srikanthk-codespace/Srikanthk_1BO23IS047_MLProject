import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



print("Step 1: Loading data...")

train = pd.read_feather("data/train.feather")
weather = pd.read_feather("data/weather_train.feather")
building = pd.read_feather("data/building_metadata.feather")

# Reduce size
train = train.sample(n=200000, random_state=42)

print("Step 2: Merging...")

df = train.merge(building, on="building_id", how="left")
df = df.merge(weather, on=["site_id", "timestamp"], how="left")

print("Step 3: Cleaning...")

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
# Forward fill (updated method)
df.ffill(inplace=True)

# Fill numeric columns only
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(0)

print("Step 4: Features...")

features = [
    'square_feet', 'year_built', 'floor_count',
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr',
    'hour', 'day', 'month'
]

X = df[features]
y = df['meter_reading']

print("Step 5: Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Step 6: Scaling...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Step 7: PCA...")

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Step 8: Training model...")

model = LinearRegression()
model.fit(X_train_pca, y_train)

print("Step 9: Predicting...")

y_pred = model.predict(X_test_pca)

print("Step 10: RMSE...")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Final RMSE:", rmse)

print("Done successfully!")

import joblib

joblib.dump(model, "output/model.pkl")
joblib.dump(pca, "output/pca.pkl")
joblib.dump(scaler, "output/scaler.pkl")

print("Model saved!")