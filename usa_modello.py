#importa le librerie necessarie
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import os, random

TF_ENABLE_ONEDNN_OPTS=0


print("tf usa la GPU:", tf.config.list_physical_devices('GPU'))

path= os.path.join(os.getcwd())
os.chdir(path)

root_dir = "datasets/flower_photos/flower_photos"


# Carico i dati dal dataset 
full_data = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    image_size=(256, 256),
    crop_to_aspect_ratio=True,
    seed=42
)

class_names = full_data.class_names

# Dividiamo il dataset in training 70%, validation15%, test15%
train_size = int(len(full_data) * 0.80) #4200
val_size = int(len(full_data) * 0.15)  #900
test_size = int(len(full_data) * 0.05)  #900

train_data = full_data.take(train_size)
test_data = full_data.skip(train_size)
val_data = test_data.skip(test_size)
test_data = test_data.take(test_size)


print(f"train_data: {len(train_data)}")
print(f"val_data: {len(val_data)}")
print(f"test_data: {len(test_data)}")


model = tf.keras.models.load_model("model_flower_256x256.keras")

for images,labels in test_data.take(1):

    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions,axis=1)

    plt.figure(figsize = (12,12))

    for i in range(min(25,len(images))):

        ax=plt.subplot(5,5,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))

        true_label = class_names[labels[i]]
        predicted_label = class_names[predicted_labels[i]]
        color = "green" if predicted_label == true_label else "red"

        plt.title(f"Pred:{predicted_label}/n (true: {true_label})", color = color, fontsize = 9)
        plt.axis ("off")

    plt.tight_layout()
    plt.show()

    break