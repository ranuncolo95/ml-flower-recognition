import tensorflow as tf                   # Importa TensorFlow
import numpy as np                        # Importa NumPy per operazioni numeriche
import matplotlib.pyplot as plt           # Importa Matplotlib per visualizzare immagini
from tensorflow.keras.preprocessing import image # Funzioni utili per immagini (non usate qui) # type: ignore
import tensorflow_datasets as tfds
import sys
import os
from PIL import Image as ImagePl          # Importa PIL per immagini (non usato qui)

path = os.path.join(os.getcwd(), "mio_modello")
os.chdir(path)
print(path)

model_path = "analisi_img.keras"    # Percorso del modello salvato
model = tf.keras.models.load_model(model_path)           # Carica il modello addestrato

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# Caricamento del dataset suddiviso
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'cifar10',
    split=[
        'train[:85%]',    # 40,000 examples (training)
        'train[85%:]',    # 10,000 examples (validation)
        'test'            # 10,000 examples (test)
    ],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Recupera i nomi delle classi
class_names = ds_info.features['label'].names

# Pre-elabora il test set (batch, normalizzazione, ecc.)
test_data = ds_test.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

# Visualizzazione delle predizioni
for images, labels in test_data.take(1):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy())
        true_label = class_names[labels[i].numpy()]
        predicted_label = class_names[predicted_labels[i]]
        color = "green" if predicted_label == true_label else "red"
        plt.title(f"Pred: {predicted_label}\n(True: {true_label})", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break