"""
Download CID Dataset and Prepare Data
"""

import os
import pathlib
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGES_URL = "https://cid-21.s3.amazonaws.com/images.tar.gz"
CSV_URL = "https://cid-21.s3.amazonaws.com/dataset.csv"

try:
    import requests
    download = lambda url, path: requests.get(url, timeout=30).content
except ImportError:
    import urllib.request
    def download(url, path):
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'curl/7.68.0')
        return urllib.request.urlopen(req).read()

def download_and_extract(url, fname, extract_to):
    filepath = f"{extract_to}/{fname}"
    extract_path = filepath.replace('.tar.gz', '')
    
    if os.path.exists(extract_path):
        print(f"✅ {fname} already exists")
        return
    
    if os.path.exists(filepath):
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(extract_to)
        os.remove(filepath)
        print(f"✅ {fname} extracted")
        return
    
    print(f"Downloading {fname}...")
    with open(filepath, 'wb') as f:
        f.write(download(url, filepath))
    
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(extract_to)
    os.remove(filepath)
    print(f"✅ {fname} ready")

# Download
print("Downloading images...")
download_and_extract(IMAGES_URL, "images.tar.gz", OUTPUT_DIR)

csv_path = f"{OUTPUT_DIR}/dataset.csv"
if not os.path.exists(csv_path):
    print("Downloading dataset.csv...")
    with open(csv_path, 'wb') as f:
        f.write(download(CSV_URL, csv_path))
    print("✅ dataset.csv ready")

# Load data
df = pd.read_csv(csv_path)
images_dir = pathlib.Path(f"{OUTPUT_DIR}/images")

# Create training dataframe
rows = []
for _, row in df.iterrows():
    for image in images_dir.glob(f"{row['sku']}/*.jpg"):
        # Skip macOS metadata files (._*)
        if image.name.startswith('._'):
            continue
        rows.append({
            'file_path': str(image),
            'teeth': row['teeth'],
            'age_in_year': row['age_in_year'],
            'breed': row['breed'],
            'height_in_inch': row['height_in_inch'],
            'weight_in_kg': row['weight_in_kg']
        })

f_df = pd.DataFrame(rows)
print(f"Total images: {len(f_df)}")

# Encode labels
f_df['teeth'] = preprocessing.LabelEncoder().fit_transform(f_df['teeth'])
f_df['breed'] = preprocessing.LabelEncoder().fit_transform(f_df['breed'])
f_df['age_in_year'] = preprocessing.LabelEncoder().fit_transform(f_df['age_in_year'])

# Filter valid images
f_df = f_df[f_df['file_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Valid images: {len(f_df)}")

# Split and save
train_df, test_df = train_test_split(f_df, test_size=0.1, random_state=42)
os.makedirs(f"{OUTPUT_DIR}/processed", exist_ok=True)
train_df.to_csv(f"{OUTPUT_DIR}/processed/train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/processed/test.csv", index=False)

print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
print("✅ Done!")
