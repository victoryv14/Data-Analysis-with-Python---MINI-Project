import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_raw_files():
    """Step 1: Load Raw Files"""
    print("="*80)
    print("STEP 1: LOADING RAW FILES")
    print("="*80)

    data_dir = "../data"
    building_metadata = pd.read_csv(os.path.join(data_dir, "building_metadata.csv"))
    weather_train = pd.read_csv(os.path.join(data_dir, "weather_train.csv"))
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), nrows=1000000)

    print(f"[+] Building Metadata shape: {building_metadata.shape}")
    print(f"[+] Weather Train shape: {weather_train.shape}")
    print(f"[+] Train shape (sampled): {train.shape}\n")

    print("Building Metadata Info:")
    print(building_metadata.info())
    print("\nTrain Data Sample:")
    print(train.head())
    print("\nWeather Data Sample:")
    print(weather_train.head())

    return building_metadata, weather_train, train

def inspect_data_quality(building_metadata, weather_train, train):
    """Step 2: Inspect Data Quality & Missing Values"""
    print("\n" + "="*80)
    print("STEP 2: INITIAL DATA QUALITY INSPECTION")
    print("="*80)

    print("\n Building Metadata Missing Values:")
    print(building_metadata.isnull().sum())
    print(f"\nPercentage of Missing Values:")
    print((building_metadata.isnull().sum() / len(building_metadata) * 100).round(2))

    print("\n Train Data Missing Values:")
    print(train.isnull().sum())

    print("\n Weather Data Missing Values:")
    print(weather_train.isnull().sum())
    print(f"\nPercentage of Missing Values:")
    print((weather_train.isnull().sum() / len(weather_train) * 100).round(2))

    print("\n Basic Statistics - Building Metadata:")
    print(building_metadata.describe())

    print("\n Basic Statistics - Train Data:")
    print(train.describe())

def align_time_bases(train, weather_train):
    """Step 3: Align Time Bases"""
    print("\n" + "="*80)
    print("STEP 3: ALIGNING TIME BASES")
    print("="*80)

    # Parse timestamps
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])

    print(f"[+] Train date range: {train['timestamp'].min()} to {train['timestamp'].max()}")
    print(f"[+] Weather date range: {weather_train['timestamp'].min()} to {weather_train['timestamp'].max()}")

    # Extract date components
    train['date'] = train['timestamp'].dt.date
    train['hour'] = train['timestamp'].dt.hour
    train['month'] = train['timestamp'].dt.month
    train['day_of_week'] = train['timestamp'].dt.dayofweek
    train['day_of_year'] = train['timestamp'].dt.dayofyear
    train['quarter'] = train['timestamp'].dt.quarter

    weather_train['date'] = weather_train['timestamp'].dt.date
    weather_train['hour'] = weather_train['timestamp'].dt.hour

    # Aggregate to daily granularity
    train_daily = train.groupby(['building_id', 'meter', 'date']).agg({
        'meter_reading': ['sum', 'mean', 'std', 'min', 'max'],
        'hour': 'count'
    }).reset_index()
    train_daily.columns = ['building_id', 'meter', 'date', 'total_energy', 'avg_energy',
                           'std_energy', 'min_energy', 'max_energy', 'readings_count']

    # Aggregate weather to daily
    weather_daily = weather_train.groupby(['site_id', 'date']).agg({
        'air_temperature': ['mean', 'min', 'max', 'std'],
        'cloud_coverage': 'mean',
        'dew_temperature': 'mean',
        'precip_depth_1_hr': 'sum',
        'sea_level_pressure': 'mean',
        'wind_direction': 'mean',
        'wind_speed': ['mean', 'max']
    }).reset_index()
    weather_daily.columns = ['site_id', 'date', 'temp_mean', 'temp_min', 'temp_max', 'temp_std',
                             'cloud_coverage', 'dew_temp', 'precip_total', 'pressure',
                             'wind_direction', 'wind_speed_mean', 'wind_speed_max']

    print(f"[+] Daily aggregated train shape: {train_daily.shape}")
    print(f"[+] Daily aggregated weather shape: {weather_daily.shape}")

    return train_daily, weather_daily

def merge_datasets(train_daily, weather_daily, building_metadata):
    """Step 4: Merge Datasets"""
    print("\n" + "="*80)
    print("STEP 4: MERGING DATASETS")
    print("="*80)

    # Merge train with metadata
    merged = train_daily.merge(building_metadata, on='building_id', how='left')
    print(f"[+] After merging with metadata: {merged.shape}")

    # Merge with weather
    merged = merged.merge(weather_daily, on=['site_id', 'date'], how='left')
    print(f"[+] After merging with weather: {merged.shape}")

    print("\nMerged Data Sample:")
    print(merged.head())

    return merged

def clean_missing_values(merged):
    """Step 5: Clean Missing Values"""
    print("\n" + "="*80)
    print("STEP 5: CLEANING MISSING VALUES")
    print("="*80)

    print("\n Missing Values Before Cleaning:")
    missing_before = merged.isnull().sum()
    print(missing_before[missing_before > 0])

    # Building metadata imputation
    merged['year_built'] = merged['year_built'].fillna(merged['year_built'].median())
    merged['floor_count'] = merged['floor_count'].fillna(1)

    # Weather features imputation (forward fill within groups, then backward fill)
    weather_cols = ['temp_mean', 'temp_min', 'temp_max', 'temp_std', 'cloud_coverage',
                    'dew_temp', 'precip_total', 'pressure', 'wind_direction',
                    'wind_speed_mean', 'wind_speed_max']

    for col in weather_cols:
        if col in merged.columns:
            merged[col] = merged.groupby('site_id')[col].transform(lambda x: x.ffill().bfill())
            merged[col] = merged[col].fillna(merged[col].median())

    # Fill remaining numeric NaNs with median
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())

    print("\n Missing Values After Cleaning:")
    missing_after = merged.isnull().sum()
    if missing_after.sum() == 0:
        print("[+] All missing values successfully handled!")
    else:
        print(missing_after[missing_after > 0])

    return merged

def handle_outliers(merged):
    """Step 6: Handle Outliers"""
    print("\n" + "="*80)
    print("STEP 6: OUTLIER DETECTION & HANDLING")
    print("="*80)

    # Detect outliers using IQR method for energy readings
    Q1 = merged['total_energy'].quantile(0.25)
    Q3 = merged['total_energy'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = ((merged['total_energy'] < lower_bound) |
                      (merged['total_energy'] > upper_bound)).sum()
    print(f"[+] Outliers detected (IQR method): {outliers_count} ({outliers_count/len(merged)*100:.2f}%)")

    # Cap outliers instead of removing
    merged['total_energy'] = merged['total_energy'].clip(lower_bound, upper_bound)

    # Similarly for temperature
    temp_Q1 = merged['temp_mean'].quantile(0.25)
    temp_Q3 = merged['temp_mean'].quantile(0.75)
    temp_IQR = temp_Q3 - temp_Q1
    merged['temp_mean'] = merged['temp_mean'].clip(
        temp_Q1 - 1.5 * temp_IQR,
        temp_Q3 + 1.5 * temp_IQR
    )

    print("[+] Outliers capped to upper/lower bounds")

    return merged

def feature_engineering(merged):
    """Step 7: Feature Engineering"""
    print("\n" + "="*80)
    print("STEP 7: FEATURE ENGINEERING")
    print("="*80)

    current_year = 2024

    # Building-related features
    merged['building_age'] = current_year - merged['year_built']
    merged['energy_per_sqft'] = merged['total_energy'] / merged['square_feet'].replace(0, np.nan)
    merged['avg_energy_per_sqft'] = merged['avg_energy'] / merged['square_feet'].replace(0, np.nan)
    merged['building_density'] = merged['floor_count'] / (merged['square_feet'] / 1000).replace(0, np.nan)

    # Weather-related features
    merged['temp_range'] = merged['temp_max'] - merged['temp_min']
    merged['temp_diff_from_dew'] = merged['temp_mean'] - merged['dew_temp']
    merged['is_high_wind'] = (merged['wind_speed_mean'] > merged['wind_speed_mean'].quantile(0.75)).astype(int)
    merged['is_rainy'] = (merged['precip_total'] > 0).astype(int)

    # Interaction features
    merged['temp_sqft_interaction'] = merged['temp_mean'] * np.log1p(merged['square_feet'])
    merged['age_energy_interaction'] = merged['building_age'] * merged['energy_per_sqft']

    # Date-based features
    merged['date'] = pd.to_datetime(merged['date'])
    merged['is_weekend'] = merged['date'].dt.dayofweek.isin([5, 6]).astype(int)
    merged['is_holiday_season'] = merged['date'].dt.month.isin([11, 12]).astype(int)

    print("[+] Features engineered:")
    print("  - Building: age, energy_per_sqft, density")
    print("  - Weather: temp_range, wind/rain indicators, interactions")
    print("  - Time: weekend, holiday_season flags")

    return merged

def encode_categorical_features(merged):
    """Step 8: Encode Categorical Features"""
    print("\n" + "="*80)
    print("STEP 8: ENCODING CATEGORICAL FEATURES")
    print("="*80)

    # Label encode primary use
    le_use = LabelEncoder()
    merged['primary_use_encoded'] = le_use.fit_transform(merged['primary_use'].astype(str))

    print(f"[+] Primary Use categories: {dict(zip(le_use.classes_, le_use.transform(le_use.classes_)))}")

    # One-hot encode meter type
    merged = pd.get_dummies(merged, columns=['meter'], prefix='meter', drop_first=False)
    print(f"[+] Meter types one-hot encoded: {[c for c in merged.columns if 'meter_' in c]}")

    return merged

def scale_numeric_features(merged):
    """Step 9: Scale Numeric Features"""
    print("\n" + "="*80)
    print("STEP 9: SCALING NUMERIC FEATURES")
    print("="*80)

    numeric_cols = ['square_feet', 'building_age', 'energy_per_sqft', 'temp_mean',
                    'temp_range', 'wind_speed_mean', 'pressure', 'building_density']
    numeric_cols = [c for c in numeric_cols if c in merged.columns]

    scaler = StandardScaler()
    merged_scaled = merged.copy()
    merged_scaled[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

    print(f"[+] Scaled features: {numeric_cols}")
    print(f"\nScaled features statistics:")
    print(merged_scaled[numeric_cols].describe().round(3))

    return merged_scaled, scaler

def visualize_data_distributions(merged, output_dir='../docs/images'):
    """Step 10: Visualize Data Distributions"""
    print("\n" + "="*80)
    print("STEP 10: DATA DISTRIBUTION VISUALIZATIONS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 10.1 Energy Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(merged['total_energy'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Distribution of Total Energy Consumption', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Energy (kWh)')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(merged['avg_energy'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_title('Distribution of Average Energy Consumption', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Energy (kWh)')
    axes[0, 1].set_ylabel('Frequency')

    # Log scale for better visualization
    axes[1, 0].hist(np.log1p(merged['total_energy']), bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('Log Distribution of Total Energy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log(Energy)')
    axes[1, 0].set_ylabel('Frequency')

    # KDE plot
    merged['total_energy'].plot.density(ax=axes[1, 1], color='purple', linewidth=2)
    axes[1, 1].set_title('Density Plot of Total Energy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Energy (kWh)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_energy_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 01_energy_distributions.png")

    # 10.2 Energy by Building Type
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    building_energy = merged.groupby('primary_use')['total_energy'].agg(['mean', 'sum', 'count'])
    building_energy_sorted = building_energy.sort_values('mean', ascending=False)

    building_energy_sorted['mean'].plot(kind='bar', ax=axes[0], color='teal', edgecolor='black')
    axes[0].set_title('Average Energy Consumption by Building Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Energy (kWh)')
    axes[0].set_xlabel('Building Type')
    axes[0].tick_params(axis='x', rotation=45)

    building_energy_sorted['count'].plot(kind='bar', ax=axes[1], color='orange', edgecolor='black')
    axes[1].set_title('Number of Records by Building Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Building Type')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_energy_by_building_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 02_energy_by_building_type.png")

    # 10.3 Box plot - Energy by Building Type
    fig, ax = plt.subplots(figsize=(14, 6))
    q_high = merged['total_energy'].quantile(0.95)
    df_filtered = merged[merged['total_energy'] < q_high]

    sns.boxplot(x='primary_use', y='total_energy', data=df_filtered, ax=ax, palette='Set2')
    ax.set_title('Energy Distribution by Building Type (Outliers Removed - 95th percentile)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Building Type')
    ax.set_ylabel('Total Energy (kWh)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_energy_boxplot_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 03_energy_boxplot_by_type.png")

    # 10.4 Weather Features Distribution
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    weather_features = ['temp_mean', 'cloud_coverage', 'wind_speed_mean', 'precip_total', 'pressure', 'temp_range']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    for idx, (ax, feature, color) in enumerate(zip(axes.flatten(), weather_features, colors)):
        ax.hist(merged[feature], bins=40, edgecolor='black', alpha=0.7, color=color)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_weather_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 04_weather_distributions.png")

    # 10.5 Temperature vs Energy Scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sample_df = merged.sample(n=min(5000, len(merged)))

    axes[0].scatter(sample_df['temp_mean'], sample_df['total_energy'], alpha=0.5, s=30, color='steelblue')
    z = np.polyfit(sample_df['temp_mean'], sample_df['total_energy'], 1)
    p = np.poly1d(z)
    axes[0].plot(sample_df['temp_mean'].sort_values(),
                p(sample_df['temp_mean'].sort_values()),
                "r--", linewidth=2, label='Trend')
    axes[0].set_title('Temperature vs. Total Energy Consumption', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mean Temperature (C)')
    axes[0].set_ylabel('Total Energy (kWh)')
    axes[0].legend()

    # Hexbin plot for density
    hb = axes[1].hexbin(sample_df['temp_mean'], sample_df['total_energy'], gridsize=20, cmap='YlOrRd')
    axes[1].set_title('Temperature vs. Energy (Density)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mean Temperature (C)')
    axes[1].set_ylabel('Total Energy (kWh)')
    plt.colorbar(hb, ax=axes[1], label='Count')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_temperature_vs_energy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 05_temperature_vs_energy.png")

    # 10.6 Energy Trends Over Time
    daily_energy = merged.groupby('date')['total_energy'].agg(['sum', 'mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily_energy['date'], daily_energy['sum'], linewidth=2, label='Total Daily Energy', color='steelblue')
    ax.fill_between(daily_energy['date'],
                     daily_energy['mean'] - daily_energy['std'],
                     daily_energy['mean'] + daily_energy['std'],
                     alpha=0.3, color='lightblue', label='1 Std Dev')
    ax.set_title('Energy Consumption Trend Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy (kWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_energy_trends_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 06_energy_trends_over_time.png")

    # 10.7 Building Age Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(merged['building_age'], merged['total_energy'], alpha=0.5, s=30, color='coral')
    axes[0].set_title('Building Age vs. Energy Consumption', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Building Age (years)')
    axes[0].set_ylabel('Total Energy (kWh)')

    merged.groupby(pd.cut(merged['building_age'], bins=10))['total_energy'].mean().plot(
        kind='bar', ax=axes[1], color='green', edgecolor='black')
    axes[1].set_title('Average Energy by Building Age Groups', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Energy (kWh)')
    axes[1].set_xlabel('Building Age Group')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_building_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 07_building_age_analysis.png")

    # 10.8 Meter Type Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    meter_cols = [c for c in merged.columns if c.startswith('meter_')]
    meter_energy = {}
    for col in meter_cols:
        meter_type = col.replace('meter_', '')
        meter_energy[meter_type] = merged[merged[col] == 1]['total_energy'].mean()

    pd.Series(meter_energy).plot(kind='bar', ax=axes[0], color='purple', edgecolor='black')
    axes[0].set_title('Average Energy by Meter Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Energy (kWh)')
    axes[0].set_xlabel('Meter Type')
    axes[0].tick_params(axis='x', rotation=0)

    merged_meter_counts = merged[meter_cols].sum()
    merged_meter_counts.index = [idx.replace('meter_', '') for idx in merged_meter_counts.index]
    merged_meter_counts.plot(kind='bar', ax=axes[1], color='gold', edgecolor='black')
    axes[1].set_title('Record Count by Meter Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Meter Type')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_meter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 08_meter_analysis.png")

    # 10.9 Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_data = merged.select_dtypes(include=[np.number]).corr()
    sns.heatmap(numeric_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix of Numeric Features', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 09_correlation_heatmap.png")

    # 10.10 Month and Day Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    monthly_data = merged.groupby(merged['date'].dt.month)['total_energy'].mean()
    monthly_data.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Average Energy Consumption by Month', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Energy (kWh)')
    axes[0].set_xlabel('Month')
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    axes[0].set_xticklabels([month_names.get(m, str(m)) for m in monthly_data.index])

    daily_data = merged.groupby(merged['date'].dt.dayofweek)['total_energy'].mean()
    daily_data.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Average Energy Consumption by Day of Week', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Energy (kWh)')
    axes[1].set_xlabel('Day of Week')
    day_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    axes[1].set_xticklabels([day_names.get(d, str(d)) for d in daily_data.index])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 10_temporal_analysis.png")

    # 10.11 Energy per Square Feet Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(merged['energy_per_sqft'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0].set_title('Distribution of Energy per Square Foot', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Energy per Sq Ft')
    axes[0].set_ylabel('Frequency')

    building_efficiency = merged.groupby('primary_use')['energy_per_sqft'].mean().sort_values(ascending=False)
    building_efficiency.plot(kind='barh', ax=axes[1], color='teal', edgecolor='black')
    axes[1].set_title('Average Energy Efficiency by Building Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Energy per Square Foot')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_energy_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 11_energy_efficiency_analysis.png")

    # 10.12 Wind Speed and Precipitation Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Wind speed vs Energy
    sample = merged.sample(n=min(3000, len(merged)))
    axes[0, 0].scatter(sample['wind_speed_mean'], sample['total_energy'], alpha=0.5, color='steelblue')
    axes[0, 0].set_title('Wind Speed vs. Energy Consumption', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Wind Speed (m/s)')
    axes[0, 0].set_ylabel('Energy (kWh)')

    # Precipitation vs Energy
    axes[0, 1].scatter(sample['precip_total'], sample['total_energy'], alpha=0.5, color='coral')
    axes[0, 1].set_title('Precipitation vs. Energy Consumption', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Precipitation (mm)')
    axes[0, 1].set_ylabel('Energy (kWh)')

    # Cloud coverage vs Energy
    axes[1, 0].scatter(sample['cloud_coverage'], sample['total_energy'], alpha=0.5, color='green')
    axes[1, 0].set_title('Cloud Coverage vs. Energy Consumption', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Cloud Coverage (%)')
    axes[1, 0].set_ylabel('Energy (kWh)')

    # Pressure vs Energy
    axes[1, 1].scatter(sample['pressure'], sample['total_energy'], alpha=0.5, color='purple')
    axes[1, 1].set_title('Sea Level Pressure vs. Energy Consumption', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Pressure (hPa)')
    axes[1, 1].set_ylabel('Energy (kWh)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_weather_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 12_weather_interactions.png")

def split_and_save_data(merged, output_dir='../data'):
    """Step 11: Split and Save Data"""
    print("\n" + "="*80)
    print("STEP 11: SPLITTING AND SAVING DATA")
    print("="*80)

    # Separate features and target
    target_col = 'total_energy'
    exclude_cols = ['total_energy', 'date', 'building_id', 'site_id', 'primary_use',
                    'timestamp', 'meter', 'avg_energy', 'std_energy', 'min_energy',
                    'max_energy', 'readings_count']

    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    X = merged[feature_cols]
    y = merged[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"[+] Training set: {X_train.shape[0]} samples")
    print(f"[+] Validation set: {X_val.shape[0]} samples")
    print(f"[+] Test set: {X_test.shape[0]} samples")

    # Save splits
    X_train.to_pickle(f'{output_dir}/X_train.pkl')
    X_val.to_pickle(f'{output_dir}/X_val.pkl')
    X_test.to_pickle(f'{output_dir}/X_test.pkl')
    y_train.to_pickle(f'{output_dir}/y_train.pkl')
    y_val.to_pickle(f'{output_dir}/y_val.pkl')
    y_test.to_pickle(f'{output_dir}/y_test.pkl')

    print("[+] Data splits saved as pickle files")

    # Save complete processed dataset
    merged.to_pickle(f'{output_dir}/merged_clean_comprehensive.pkl')
    print("[+] Complete processed dataset saved")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_summary_report(merged):
    """Step 12: Create Summary Report"""
    print("\n" + "="*80)
    print("STEP 12: SUMMARY REPORT")
    print("="*80)

    print("\n DATASET DIMENSIONS:")
    print(f"Total records: {len(merged):,}")
    print(f"Total features: {merged.shape[1]}")

    print("\n ENERGY CONSUMPTION STATISTICS:")
    print(f"Total Energy - Mean: {merged['total_energy'].mean():.2f} kWh")
    print(f"Total Energy - Median: {merged['total_energy'].median():.2f} kWh")
    print(f"Total Energy - Std Dev: {merged['total_energy'].std():.2f} kWh")
    print(f"Total Energy - Min: {merged['total_energy'].min():.2f} kWh")
    print(f"Total Energy - Max: {merged['total_energy'].max():.2f} kWh")

    print("\n BUILDING TYPES:")
    print(merged['primary_use'].value_counts())

    print("\n TEMPERATURE STATISTICS:")
    print(f"Temp Mean - Range: {merged['temp_mean'].min():.2f}C to {merged['temp_mean'].max():.2f}C")
    print(f"Temp Mean - Average: {merged['temp_mean'].mean():.2f}C")

    print("\n FEATURE ENGINEERING RESULTS:")
    print(f"Features created: {merged.shape[1]}")
    print(f"Numeric features: {merged.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Categorical features: {merged.select_dtypes(include=['object']).shape[1]}")

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE ENERGY CONSUMPTION DATA ANALYSIS PIPELINE")
    print("="*80 + "\n")

    # Execute all steps
    building_metadata, weather_train, train = load_raw_files()
    inspect_data_quality(building_metadata, weather_train, train)
    train_daily, weather_daily = align_time_bases(train, weather_train)
    merged = merge_datasets(train_daily, weather_daily, building_metadata)
    merged = clean_missing_values(merged)
    merged = handle_outliers(merged)
    merged = feature_engineering(merged)
    merged = encode_categorical_features(merged)
    merged_scaled, scaler = scale_numeric_features(merged)
    visualize_data_distributions(merged)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_data(merged)
    create_summary_report(merged)

    print("\n" + "="*80)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
