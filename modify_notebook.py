import nbformat
import sys

notebook_path = r'C:\Users\Vamsi Kalyan\OneDrive\Desktop\III SEM\DAE\Data-Analysis-with-Python---MINI-Project\notebooks\01_energy_analysis_pipeline.ipynb'

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Define new encoding cell source (from analysis_pipeline.py lines 70-76)
new_encoding_source = '''# Step 6: Encoding Categoricals...
print("Step 6: Encoding Categoricals...")
le = LabelEncoder()
merged['primary_use_encoded'] = le.fit_transform(merged['primary_use'].astype(str))
merged = pd.get_dummies(merged, columns=['meter'], prefix='meter')
'''

# Define new scaling cell source (adapted from comprehensive_pipeline.py scaling step)
new_scaling_source = '''# Step 7: Scaling Numeric Features...
print("Step 7: Scaling Numeric Features...")
numeric_cols = ['square_feet', 'building_age', 'energy_per_sqft', 'temp_mean',
                'temp_range', 'wind_speed_mean', 'pressure', 'building_density']
numeric_cols = [c for c in numeric_cols if c in merged.columns]

scaler = StandardScaler()
merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols].fillna(0))

print(f"[+] Scaled features: {numeric_cols}")
'''

# Find and replace encoding cell (cell id "da6e274d")
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.id == 'da6e274d':
        cell.source = new_encoding_source
        break

# Find and replace scaling cell (cell id "42c1b5a0")
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.id == '42c1b5a0':
        cell.source = new_scaling_source
        break

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")