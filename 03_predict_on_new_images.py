"""
Make Predictions on New Images
"""

import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("STEP 3: MAKE PREDICTIONS")
print("="*60 + "\n")

IMG_SIZE = 150
MODEL_PATH = "cattle_trait_model.h5"

# Load model
print(f"📂 Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"   ✅ {MODEL_PATH} loaded")

# Prediction function
def predict_traits(image_path):
    """Predict all traits for single image"""
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = keras.preprocessing.image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    w_pred, h_pred, a_pred, b_pred = model.predict(img_arr, verbose=0)
    
    return {
        'weight': w_pred[0][0],
        'height': h_pred[0][0],
        'age_class': np.argmax(a_pred[0]),
        'age_conf': np.max(a_pred[0]),
        'breed_class': np.argmax(b_pred[0]),
        'breed_conf': np.max(b_pred[0])
    }

# Predict on test images
print(f"\n🔍 Making predictions on test images...")
test_df = pd.read_csv("data/processed/test.csv")

# Filter out invalid images (._* files and non-existent files)
test_df = test_df[test_df['file_path'].apply(lambda x: os.path.exists(x) and not os.path.basename(x).startswith('._'))].reset_index(drop=True)

count = 0
i = 0
while count < 5 and i < len(test_df):
    image_path = test_df.iloc[i]['file_path']
    
    # Skip if it's a metadata file
    if os.path.basename(image_path).startswith('._'):
        i += 1
        continue
    
    print(f"\n   Image {count+1}: {os.path.basename(image_path)}")
    
    try:
        result = predict_traits(image_path)
        print(f"      Weight: {result['weight']:.2f} kg")
        print(f"      Height: {result['height']:.2f} inches")
        print(f"      Age class: {result['age_class']} (conf: {result['age_conf']:.1%})")
        print(f"      Breed class: {result['breed_class']} (conf: {result['breed_conf']:.1%})")
        count += 1
    except Exception as e:
        print(f"      ❌ Error: {e}")
    finally:
        i += 1

print("\n" + "="*60)
print("✅ PREDICTIONS COMPLETE!")
print("="*60 + "\n")
