import tensorflow_datasets as tfds
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import os, random
import PIL.Image
import pathlib
import PIL


TF_ENABLE_ONEDNN_OPTS=0


# importiamo il dataset flower_photos
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True,
                                   cache_dir='.') # questa parte lo scarica dentro la current directory


# analisi del dataset
# conteggio di quante immagini ci sono in ogni classe 

counts = {}
data_dir = os.path.join(os.getcwd(), "datasets", "flower_photos","flower_photos")

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

h_shape = np.array([])
w_shape = np.array([])

for classe in os.listdir(data_dir):
    if not classe.endswith('.txt'): 
        for i in os.listdir(os.path.join(data_dir, classe)):
            path = os.path.join(data_dir, classe, i)
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            h_shape = np.append(h_shape, h)
            w_shape = np.append(w_shape, w) 


df_shape = pd.DataFrame({"h" : h_shape, "w": w_shape})

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
    image_size= (256,256),
    crop_to_aspect_ratio = True,
    seed=42
)


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

train_data_prefetched = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data_prefetched = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data_prefetched = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# crea modello
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256,256,3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(len(full_data.class_names), activation="softmax")
])

model.summary()

model.compile(
    optimizer= "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

training = model.fit(
    train_data_prefetched,
    validation_data = val_data_prefetched,
    epochs = 15
)


# salva il modello
model.save("model_flower_256x256.keras")

# valuto il modello

pd.DataFrame(training.history)[["accuracy", "val_accuracy"]].plot()
plt.title("accuracy e val accuracy per epoch")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

