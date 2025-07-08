from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from defs.defs import *
import seaborn as sns
import pandas as pd
import numpy as np
import os, random
import PIL.Image
import pathlib
import PIL


root_dir = os.getcwd()
os.chdir(root_dir)                                        
data_dir = os.path.join(root_dir, "datasets", "flower_photos","flower_photos")

batch_size = 32                         # Numero di immagini per batch che analizza per aggiornare i pesi del modello
img_height = 256                         # Altezza delle immagini
img_width = 256                         # Larghezza delle immagini

# importiamo il dataset flower_photos
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True,
                                   cache_dir='.') # questa parte lo scarica dentro la current directory


# analisi del dataset
# conteggio di quante immagini ci sono in ogni classe 

counts = {}

for folder in os.listdir(data_dir):
    path = os.path.join(data_dir, folder)
    if os.path.isdir(path):
        num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])
        counts[folder] = num_files

print(counts) # print del count del dataset

df = pd.DataFrame.from_dict(counts, orient="index", columns=["num_images"])
df.index.name = "categoria"
df = df.sort_values("num_images", ascending=False)

plt.figure(figsize=(8, len(df) * 0.6))
sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
plt.title("Numero di immagini per categoria")
plt.tight_layout()
plt.show()


# analisi shape immagini

# 1. Create a TensorFlow dataset from paths
def get_image_dims(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.shape(img)[:2]  # Returns (height, width)

# Get all image paths
image_paths = []
for classe in os.listdir(data_dir):
    if not classe.endswith('.txt'):
        class_dir = os.path.join(data_dir, classe)
        image_paths.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Convert to TensorFlow dataset
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
dims_ds = path_ds.map(get_image_dims, num_parallel_calls=tf.data.AUTOTUNE)

# Get all dimensions
sizes = list(dims_ds.as_numpy_iterator())
h_shape, w_shape = zip(*sizes)

# Create DataFrame
df_shape = pd.DataFrame({"h": h_shape, "w": w_shape})
print(df_shape.describe())


#print(df_shape.head())
sns.jointplot(data=df_shape, x="w", y="h", kind="scatter")
plt.show()


# visualizza alcune immagini
plt.figure(figsize=(12,12))
for i in range(25):
    def choose_random_img(data_dir):
        random_folder = random.choice(os.listdir(data_dir))

        if not random_folder.endswith('.txt'): 
            file_name = random.choice(os.listdir(os.path.join(data_dir, random_folder)))
            path = os.path.join(data_dir, random_folder, file_name)
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            #print(f"shape: {h,w}")
            return plt.subplot(5,5, i+1), plt.imshow(img), plt.axis("off"), plt.title(random_folder)
        
        else:
            choose_random_img(data_dir)

    choose_random_img(data_dir)
plt.show()


# caricato in full_data il dataset di immagini compreso di classi
full_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    crop_to_aspect_ratio = True,
    seed = 42
)

'''
# Recommended augmentation (add this before model definition)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Safe for flowers
    layers.RandomRotation(0.05),     # ±18° (more natural for flowers)
    layers.RandomZoom(0.1, 0.1),     # Slight zoom (10% max)
    layers.RandomContrast(0.1),      # Mild contrast adjustment
    layers.RandomBrightness(0.1),    # Subtle brightness changes
    layers.RandomTranslation(        # Gentle shifts
        height_factor=0.05,
        width_factor=0.05,
        fill_mode='reflect'          # Better edge handling
    ),
])
'''

train_size = int(len(full_data) * 0.8)
val_size = int(len(full_data) * 0.15)
test_size = int(len(full_data) * 0.05)


# passaggio più performante
train_data = full_data.take(train_size)
test_data = full_data.skip(train_size)
val_data = test_data.skip(test_size)
test_data = test_data.take(test_size)

print(f" train data: {len(train_data)}")
print(f" test data: {len(test_data)}")
print(f" val data: {len(val_data)}")

'''
# ★★★ ADD PREPROCESSING PIPELINES HERE ★★★
def preprocess_train(image, label):
    image = data_augmentation(image, training=True)  # Augment only training
    image = tf.cast(image, tf.float32) / 255.0       # Normalize
    return image, label

def preprocess_val_test(image, label):
    image = tf.cast(image, tf.float32) / 255.0       # Only normalize
    return image, label


# Apply preprocessing
train_data = train_data.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
val_data = val_data.map(preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.map(preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE)


# Check augmented training samples
for images, _ in train_data.take(1):
    plt.imshow(images[0].numpy())
    plt.title("Augmented Training Sample")
    plt.show()

# Check raw validation samples
for images, _ in val_data.take(1):
    plt.imshow(images[0].numpy())
    plt.title("Unaugmented Validation Sample")
    plt.show()
'''

# optimize 
train_data_prefetched = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data_prefetched = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data_prefetched = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

tf.keras.backend.clear_session()        

# crea modello
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    # data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    #tf.keras.layers.Conv2D(64, 3, activation="relu"),
    #tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Conv2D(32, 3, activation="relu"),
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(len(full_data.class_names), activation="softmax")
])

model.summary()

model.compile(
    optimizer= "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

tensorboard_callback = TensorBoard()  # Callback per TensorBoard

# Addestra il modello
early_stopping = EarlyStopping(
    monitor ='val_accuracy',                        # Monitora l'accuratezza di validazione
    patience = 5,                                   # Numero di epoche senza miglioramento prima di fermare l'addestramento
    restore_best_weights=True                       # Ripristina i pesi del modello alla migliore versione trovata
)


callbacks = [early_stopping, TrainingMonitor(), tensorboard_callback]

history = model.fit(
    train_data_prefetched,                                # Dati di addestramento
    epochs = 1,                                    # Numero di epoche per l'addestramento
    validation_data=val_data_prefetched,           # Dati di validazione
    callbacks = callbacks,                     # Callback per l'early stopping
    shuffle=False                                   # Non mescola i dati durante l'addestramento
)

# analyze model's training
analyze_training(history)

# salva il modello
model.save("model_flower_256x256_light.keras")

# valuto il modello
# Creo le variabili di appoggio per i dati di addestramento e validazione
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']


# Convert history to DataFrame
history_df = pd.DataFrame(history.history)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss on first subplot
history_df[["loss", "val_loss"]].plot(
    ax=ax1,
    color=['blue', 'orange']
)
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Plot Accuracy on second subplot
history_df[["accuracy", "val_accuracy"]].plot(
    ax=ax2,
    color=['green', 'red']
)
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.show()