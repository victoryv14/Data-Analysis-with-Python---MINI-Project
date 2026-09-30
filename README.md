# Energy Consumption Across Buildings

## Problem Statement
Buildings consume around 30% of the world's energy, yet most of that 
consumption happens without any measurement or optimization. This project 
uses data science to analyze and predict energy consumption patterns 
across different building types.

## Objectives
1. Identify key factors driving energy consumption across buildings
2. Reveal hidden patterns and anomalies in energy usage
3. Quantify consumption differences across building categories
4. Build a model to accurately predict building energy consumption
5. Provide actionable insights to reduce energy waste

## Detailed Project Workflow

1. **Data Acquisition & Initial Inspection**: 
   - Loaded raw American Society of Heating, Refrigerating and Air-Conditioning Engineers datasets: energy usage (`train.csv`), building characteristics (`building_metadata.csv`), and weather patterns (`weather_train.csv`). 
   - Conducted initial exploratory data analysis to identify data distributions, scales, and missing values.

2. **Time-Series Alignment & Aggregation**:
   - Synchronized disparate temporal recordings from energy and weather data.
   - Aggregated raw hourly energy data to daily granularity, reducing computational load and smoothing out volatile usage noise.

3. **Integrated Data Merging**:
   - Built a comprehensive unified dataset by merging energy, metadata, and daily-averaged weather data using relational keys (`building_id`, `site_id`, `date`).

4. **Data Cleaning & Imputation**:
   - Systematically handled missing values:
     - Imputed building metadata (`year_built`, `floor_count`) with median/assumed values.
     - Implemented group-based forward and backward filling for weather variables within each `site_id` to maintain local consistency.

5. **Outlier Detection & Handling**:
   - Employed the Interquartile Range method to identify extreme energy and temperature anomalies. 
   - Capped outliers within the $1.5 \times$ Interquartile Range range to reduce the influence of extreme anomalies while retaining data points.

6. **Feature Engineering**:
   - Engineered domain-specific features to capture nuanced behavior:
     - **Building Metrics**: `building_age` (current year - `year_built`), `energy_per_sqft` (efficiency indicator), and `building_density`.
     - **Weather Indicators**: `temp_range`, `is_high_wind`, and `is_rainy` to capture environment impacts.
     - **Temporal Flags**: `is_weekend` and `is_holiday_season` to identify cyclical, user-driven usage patterns.

7. **Data Transformation**:
   - Applied `LabelEncoder` for `primary_use` categories and One-Hot encoding for `meter` types.
   - Standardized numerical features using `StandardScaler` to ensure all inputs contribute proportionately to model training.

8. **Visualization & Analytics**:
   - Generated distributions and correlations to visualize usage patterns, seasonality, and the impact of meteorological variables on energy demand.

## Key Insights

- **Consumption Variation**: Energy usage varies significantly by building type and size, with clear patterns emerging after normalizing by square footage (`energy_per_sqft`).
- **Weather Sensitivity**: Heating, Ventilation, and Air Conditioning operations drive significant energy consumption, with building demand showing high correlation to `temp_mean` and `dew_temp` variances.
- **Dimensional Patterns**: Aggregating data at the daily level effectively revealed long-term trends and seasonality versus short-term usage volatility.
- **Categorical Impact**: Building usage types and temporal indicators (`is_weekend` / `is_holiday_season`) are strong predictors of energy behavior, highlighting distinct operational profiles during office hours, weekends, and seasonal transitions.

## Visualization Interpretation

- **Line Chart (Monthly Trend)**: Shows clear seasonal patterns with peak consumption during summer months (July-August) due to cooling demands and lower consumption in winter months, highlighting the importance of HVAC optimization.
- **Bar Chart (Building Comparison)**: Reveals that certain building types (e.g., public assembly, education) consume significantly more energy per square foot than others, indicating targeted efficiency opportunities.
- **Pie Chart (Energy Source)**: Demonstrates that electricity is the dominant energy source (~60%), followed by chilled water and steam, suggesting that electrical efficiency measures would yield the highest impact.
- **Additional Visualizations**: Correlation heatmaps show strong relationships between temperature and energy consumption, while boxplots reveal outliers that may indicate operational inefficiencies or metering issues.

## Dataset
American Society of Heating, Refrigerating and Air-Conditioning Engineers Great Energy Predictor III — Kaggle
https://www.kaggle.com/c/ashrae-energy-prediction

## Project Structure
- /data → datasets
- /notebooks → Jupyter notebooks for analysis
- /docs → reports, charts, and documentation
- /src → Python scripts and pipelines

## Team
| Member | Role |
|--------|------|
| Pasala Harini | Problem Lead |
| Laveti Roshini | Data Analyst |
| Muppanaboina Vamsi Kalyan | Data Engineer |
| Palivela Praharsha Sai Charan | DevOps / Docs |

## Enhancements for Energy Optimization

Based on the analysis conducted, the following enhancements are recommended for energy optimization:

1. **HVAC System Optimization**: Since weather sensitivity analysis shows strong correlation between temperature and energy consumption, implementing smart HVAC controls that adjust based on occupancy and weather forecasts could reduce energy usage by 15-25%.

2. **Building-Specific Efficiency Programs**: Target buildings with highest energy_per_sqft values (identified in bar chart analysis) for retrofits including LED lighting, improved insulation, and energy-efficient appliances.

3. **Load Shifting Strategies**: Analyze daily and weekly patterns to shift non-essential energy consumption to off-peak hours, taking advantage of lower utility rates and reducing grid stress.

4. **Sub-metering and Monitoring**: Install additional sub-metering to identify specific energy drains within buildings and enable real-time anomaly detection.

5. **Renewable Energy Integration**: Given electricity's dominant share in the energy pie chart, consider solar panel installations or renewable energy procurement for high-consumption buildings.

6. **Predictive Maintenance**: Use the trained models to predict equipment failures before they occur, preventing energy waste from inefficient operation.

7. **Behavioral Intervention Programs**: Develop occupant education programs based on temporal patterns (weekend/holiday effects) to encourage energy-saving behaviors.

## Setup
```bash
pip install -r requirements.txt
```
