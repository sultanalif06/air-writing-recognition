import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob
import os

extract_dir = ".\Dataset airwriting"  


target_points = 50  
min_points_threshold = 20 

all_files = glob.glob(os.path.join(extract_dir, "*.csv"))

preprocessed_data = []

for file_path in all_files:
    # Ekstrak label dari nama file
    filename = os.path.basename(file_path)
    label = filename.split('_')[1]

    # Baca file gesture
    df_gesture = pd.read_csv(file_path)
    points = df_gesture[['x', 'y']].values

    if len(points) < min_points_threshold:
        continue 

    # Resample ke jumlah titik target
    indices = np.linspace(0, len(points) - 1, target_points)
    x_interp = interp1d(np.arange(len(points)), points[:, 0], kind='linear')(indices)
    y_interp = interp1d(np.arange(len(points)), points[:, 1], kind='linear')(indices)

    # Centering 
    x_centered = x_interp - np.mean(x_interp)
    y_centered = y_interp - np.mean(y_interp)

    # Scaling 
    max_range = max(np.ptp(x_centered), np.ptp(y_centered)) + 1e-8
    x_scaled = x_centered / max_range
    y_scaled = y_centered / max_range

    feature_vector = np.concatenate([x_scaled, y_scaled])
    preprocessed_data.append((feature_vector, label))

# Konversi ke dataframe final
features = np.array([f for f, l in preprocessed_data])
labels = np.array([l for f, l in preprocessed_data])
features_df = pd.DataFrame(features)
features_df['label'] = labels

# Simpan hasil akhir ke CSV 
features_df.to_csv("Preprocessed_Airwriting_v2.csv", index=False)

print("✅ Preprocessing selesai!")
print("Jumlah total gesture:", features_df.shape[0])
print("Jumlah fitur per gesture:", features_df.shape[1]-1)
