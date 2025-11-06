"""
Build and Train Multi-Output Cattle Model
"""

import keras
from keras import layers, Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("STEP 2: BUILD AND TRAIN MODEL")
print("="*60 + "\n")

# Config
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Load data
print("📂 Loading data...")
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
print(f"   Train: {len(train_df)}")
print(f"   Test: {len(test_df)}")

# Load images
def load_images(df, img_size=IMG_SIZE):
    """Load all images into memory, skipping corrupted ones"""
    print(f"\n🖼️  Loading {len(df)} images...")
    images = []
    valid_indices = []
    
    for idx, path in enumerate(df['file_path']):
        try:
            img = keras.preprocessing.image.load_img(path, target_size=(img_size, img_size))
            img_arr = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_arr)
            valid_indices.append(idx)
        except Exception as e:
            print(f"   ⚠️  Skipping corrupted image: {path}")
            continue
    
    print(f"   ✅ Loaded {len(images)}/{len(df)} images ({len(df) - len(images)} corrupted)")
    return np.array(images), valid_indices

X_train, train_valid_idx = load_images(train_df)
X_test, test_valid_idx = load_images(test_df)

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Filter labels to match valid images
train_df_valid = train_df.iloc[train_valid_idx].reset_index(drop=True)
test_df_valid = test_df.iloc[test_valid_idx].reset_index(drop=True)

# Prepare outputs
print(f"\n📊 Preparing labels...")
y0_train = train_df_valid['weight_in_kg'].values.astype(np.float32)
y1_train = train_df_valid['height_in_inch'].values.astype(np.float32)
y2_train = keras.utils.to_categorical(train_df_valid['age_in_year'])
y3_train = keras.utils.to_categorical(train_df_valid['breed'])

y0_test = test_df_valid['weight_in_kg'].values.astype(np.float32)
y1_test = test_df_valid['height_in_inch'].values.astype(np.float32)
y2_test = keras.utils.to_categorical(test_df_valid['age_in_year'])
y3_test = keras.utils.to_categorical(test_df_valid['breed'])

print(f"   Weight: {y0_train.shape}")
print(f"   Height: {y1_train.shape}")
print(f"   Age (classes): {y2_train.shape}")
print(f"   Breed (classes): {y3_train.shape}")

# Build model
print(f"\n🏗️  Building model...")
input_img = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img")

# CNN
x = layers.Conv2D(32, 3, activation="relu")(input_img)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.GlobalMaxPooling2D()(x)

# Dense
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Outputs
out_weight = layers.Dense(1, activation='linear', name='weight_reg')(x)
out_height = layers.Dense(1, activation='linear', name='height_reg')(x)
out_age = layers.Dense(y2_train.shape[1], activation='softmax', name='age_cls')(x)
out_breed = layers.Dense(y3_train.shape[1], activation='softmax', name='breed_cls')(x)

model = Model(inputs=input_img, outputs=[out_weight, out_height, out_age, out_breed])

# Compile
print(f"\n⚙️  Compiling...")
model.compile(
    loss={
        'weight_reg': 'mse',
        'height_reg': 'mse',
        'age_cls': 'categorical_crossentropy',
        'breed_cls': 'categorical_crossentropy'
    },
    metrics={
        'weight_reg': 'mse',
        'height_reg': 'mse',
        'age_cls': 'accuracy',
        'breed_cls': 'accuracy'
    },
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
)

model.summary()

# Train
print(f"\n🚀 Training {EPOCHS} epochs...")
history = model.fit(
    X_train,
    [y0_train, y1_train, y2_train, y3_train],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# Evaluate
print(f"\n📊 Evaluating...")
test_results = model.evaluate(X_test, [y0_test, y1_test, y2_test, y3_test], verbose=1)
print(f"\n✅ Test Loss: {test_results[0]:.4f}")

# Save
print(f"\n💾 Saving model...")
model.save("cattle_trait_model.h5")
print(f"   ✅ cattle_trait_model.h5")

# Plot
print(f"\n📈 Plotting...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(history.history['weight_reg_loss'], label='Train')
axes[0, 0].plot(history.history['val_weight_reg_loss'], label='Val')
axes[0, 0].set_title('Weight Loss')
axes[0, 0].legend()

axes[0, 1].plot(history.history['height_reg_loss'], label='Train')
axes[0, 1].plot(history.history['val_height_reg_loss'], label='Val')
axes[0, 1].set_title('Height Loss')
axes[0, 1].legend()

axes[1, 0].plot(history.history['age_cls_accuracy'], label='Train')
axes[1, 0].plot(history.history['val_age_cls_accuracy'], label='Val')
axes[1, 0].set_title('Age Accuracy')
axes[1, 0].legend()

axes[1, 1].plot(history.history['breed_cls_accuracy'], label='Train')
axes[1, 1].plot(history.history['val_breed_cls_accuracy'], label='Val')
axes[1, 1].set_title('Breed Accuracy')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print(f"   ✅ training_history.png")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nNext: python 03_predict_on_new_images.py\n")
