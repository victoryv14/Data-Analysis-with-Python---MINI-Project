import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def run_pipeline():
    # 1. Load Raw Files
    print("Step 1: Loading raw files...")
    data_dir = "../data"
    building_metadata = pd.read_csv(os.path.join(data_dir, "building_metadata.csv"))
    weather_train = pd.read_csv(os.path.join(data_dir, "weather_train.csv"))

    # We load a chunk of train data to keep memory usage manageable on standard machines
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), nrows=500000)

    print(f"Building Metadata shape: {building_metadata.shape}")
    print(f"Weather Train shape: {weather_train.shape}")
    print(f"Train shape (sampled): {train.shape}")

    # 2. Align Time Bases
    print("Step 2: Aligning time bases...")
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])

    # Aggregate energy data to daily granularity to reduce data size and noise
    train['date'] = train['timestamp'].dt.date
    train_daily = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()

    weather_train['date'] = weather_train['timestamp'].dt.date
    # Average weather features per day
    weather_daily = weather_train.groupby(['site_id', 'date']).mean(numeric_only=True).reset_index()

    # 3. Merge Datasets
    print("Step 3: Merging datasets...")
    merged = train_daily.merge(building_metadata, on='building_id', how='left')
    merged = merged.merge(weather_daily, on=['site_id', 'date'], how='left')

    print(f"Merged Dataset Shape: {merged.shape}")

    # 4. Clean Missing Values
    print("Step 4: Cleaning missing values...")
    # Process Categorical Missing Values
    merged['year_built'] = merged['year_built'].fillna(merged['year_built'].median())
    merged['floor_count'] = merged['floor_count'].fillna(1) # Assumption for missing floors

    # Process Numeric Missing Values from weather
    weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    for col in weather_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    # Drop rows where target is missing
    merged = merged.dropna(subset=['meter_reading'])

    # 5. Feature Engineering
    print("Step 5: Feature Engineering...")
    current_year = 2024
    merged['building_age'] = current_year - merged['year_built']
    # Handle potential division by zero
    merged['energy_per_sqft'] = merged['meter_reading'] / merged['square_feet'].replace(0, np.nan)

    # Extract date features
    merged['date'] = pd.to_datetime(merged['date'])
    merged['month'] = merged['date'].dt.month
    merged['day_of_week'] = merged['date'].dt.dayofweek

    # 6. Encode Categoricals
    print("Step 6: Encoding Categoricals...")
    le = LabelEncoder()
    merged['primary_use_encoded'] = le.fit_transform(merged['primary_use'].astype(str))

    # Create dummy variables for meter type - handling memory effectively
    merged = pd.get_dummies(merged, columns=['meter'], prefix='meter')

    # 7. Scale Numeric Features
    print("Step 7: Scaling Numeric Features...")
    num_cols = ['square_feet', 'air_temperature', 'building_age', 'energy_per_sqft']
    # Only scale columns that exist
    num_cols = [c for c in num_cols if c in merged.columns and not merged[c].isnull().all()]

    scaler = StandardScaler()
    # Fill remaining NaNs before scaling
    merged[num_cols] = merged[num_cols].fillna(0)
    merged[num_cols] = scaler.fit_transform(merged[num_cols])

    # 8. Split Data
    print("Step 8: Splitting Data for modeling...")
    features = [c for c in merged.columns if c not in ['meter_reading', 'date', 'building_id', 'site_id', 'primary_use', 'timestamp']]
    X = merged[features]
    y = merged['meter_reading']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # 9. Visualize Key Insights
    print("Step 9: Visualizing Key Insights...")
    os.makedirs('../docs/images', exist_ok=True)

    # 9.1 Histogram of energy
    plt.figure(figsize=(10,6))
    sns.histplot(merged['meter_reading'], bins=50, kde=True)
    plt.title('Distribution of Energy Consumption (Meter Reading)')
    plt.xlabel('Energy (kWh / equivalent)')
    plt.xlim(0, merged['meter_reading'].quantile(0.95))
    plt.savefig('../docs/images/energy_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9.2 Bar chart of average energy by building type
    plt.figure(figsize=(12,6))
    agg = merged.groupby('primary_use')['meter_reading'].mean().sort_values(ascending=False)
    agg.plot.bar()
    plt.title('Average Energy Consumption by Building Type')
    plt.ylabel('Average Energy')
    plt.xlabel('Building Primary Use')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('../docs/images/avg_energy_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9.3 Box plot of energy across primary use
    plt.figure(figsize=(14,6))
    q_high = merged['meter_reading'].quantile(0.95)
    df_filtered = merged[merged['meter_reading'] < q_high]
    sns.boxplot(x='primary_use', y='meter_reading', data=df_filtered)
    plt.title('Energy Distribution by Primary Use (Outliers Removed)')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('../docs/images/energy_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9.4 Scatter with regression line: temperature vs. energy
    plt.figure(figsize=(10,6))
    sample_df = merged.sample(n=min(5000, len(merged)))
    sns.regplot(x='air_temperature', y='meter_reading', data=sample_df, line_kws={'color':'red'})
    plt.title('Temperature vs. Energy Consumption')
    plt.xlabel('Air Temperature (Scaled)')
    plt.ylabel('Energy Consumption')
    plt.savefig('../docs/images/temp_vs_energy_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Save Processed Output
    print("Step 10: Saving Processed Output...")
    merged.to_pickle(os.path.join(data_dir, 'merged_clean.pkl'))
    print("Done! Data saved to 'data/merged_clean.pkl' and charts to 'docs/images/'")

if __name__ == "__main__":
    run_pipeline()
