import nbformat

notebook_path = r'C:\Users\Vamsi Kalyan\OneDrive\Desktop\III SEM\DAE\Data-Analysis-with-Python---MINI-Project\notebooks\01_energy_analysis_pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Encoding cell (replace cell id "da6e274d")
encoding_source = '''# 6. Encode Categoricals
print("Step 6: Encoding Categoricals...")
le = LabelEncoder()
merged['primary_use_encoded'] = le.fit_transform(merged['primary_use'].astype(str))

# Create dummy variables for meter type - handling memory effectively
merged = pd.get_dummies(merged, columns=['meter'], prefix='meter')
'''

# Scaling cell (replace cell id "42c1b5a0")
scaling_source = '''# 7. Scale Numeric Features
print("Step 7: Scaling Numeric Features...")
numeric_cols = ['square_feet', 'building_age', 'energy_per_sqft', 'temp_mean',
                'temp_range', 'wind_speed_mean', 'pressure', 'building_density']
numeric_cols = [c for c in numeric_cols if c in merged.columns]

scaler = StandardScaler()
merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

print(f"[+] Scaled features: {numeric_cols}")
print(f"\nScaled features statistics:")
print(merged[numeric_cols].describe().round(3))
'''

for cell in nb.cells:
    if cell.cell_type == 'code' and cell.id == 'da6e274d':
        cell.source = encoding_source
    if cell.cell_type == 'code' and cell.id == '42c1b5a0':
        cell.source = scaling_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")