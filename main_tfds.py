# importo librerie necessarie e modelli pretrainati
import tensorflow as tf
import tensorflow_datasets as tfds        # Importa TensorFlow Datasets
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Suddivide: 80% del train ufficiale come train, 15%+5% del test ufficiale come val e test
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
    'mnist',
    split=[
        'train[:80%]',    # 40,000 examples (training)
        'train[80%:]',    # 10,000 examples (validation)
        'test'            # 10,000 examples (test)
    ],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
    class_names = ds_info.features['label'].names
    print(f"Numero di classi dentro al dataset: {len(class_names)}")

    # Inizializza il dizionario dei conteggi
    label_counts = {i: 0 for i in range(len(class_names))}

    # Conta le occorrenze
    for image, label in tfds.as_numpy(ds_train):
        label_counts[label] += 1

    # Mappa da indice a nome della classe
    counts_by_class = {class_names[i]: label_counts[i] for i in range(len(class_names))}

    # Stampa il risultato
    print(counts_by_class)

    # creazione del dataframe con le analisi dei dati
    df = pd.DataFrame.from_dict(counts_by_class, orient='index', columns=['num_images'])
    df.index.name = 'categoria'
    df = df.sort_values('num_images', ascending=False)

    # crea heatmap e lo visualizza
    plt.figure(figsize=(8, len(df*0.6)))
    sns.heatmap(df,annot=True, fmt='d', cmap='Blues')
    plt.title("Numero di immagini per categoria")
    plt.tight_layout()
    plt.show()


    def normalize_image(img, label):
        return tf.cast(img, tf.float32) / 255., label   # Normalizza le immagini tra 0 e 1

    # Prepara il dataset di training: normalizza, memorizza in cache, mescola, crea batch e prefetch
    ds_train = ds_train.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepara il dataset di test: normalizza, crea batch, memorizza in cache e prefetch
    ds_val = ds_val.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
    ds_val = ds_val.batch(128)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    tensorboard_callback = TensorBoard()  # Callback per TensorBoard

    # Definisce il modello sequenziale: Flatten, Dense, Dropout, Dense finale
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)),   # Normalizza i pixel tra 0 e 1; input: immagini 32x32 RGB
        tf.keras.layers.Conv2D(64, 5, activation='relu'),             # Convoluzione: 6 filtri, kernel 5x5, attivazione ReLU
        tf.keras.layers.MaxPooling2D(),                               # Pooling massimo 2x2 (default)
        tf.keras.layers.Dropout(0.2),                                 # Dropout: disattiva il 10% dei neuroni (riduce overfitting)
        tf.keras.layers.Conv2D(32, 5, activation='relu'),             # Seconda convoluzione: 16 filtri, kernel 5x5, ReLU
        tf.keras.layers.MaxPooling2D(),                               # Altro pooling massimo 2x2
        tf.keras.layers.Dropout(0.2),                                 # Dropout: disattiva il 20% dei neuroni
        tf.keras.layers.Flatten(),                                    # Appiattisce i dati in un vettore 1D
        tf.keras.layers.Dense(128, activation='relu'),                # Strato denso: 128 neuroni, attivazione ReLU
        tf.keras.layers.Dense(len(class_names), activation='softmax') # Output: un neurone per classe, softmax per probabilità
    ])

    model.summary()   # Mostra un riepilogo del modello

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)  # Funzione di perdita

    # Compila il modello con ottimizzatore Adam, funzione di perdita e accuratezza come metrica
    model.compile(
        optimizer = 'adam',
        loss = loss_fn,
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Allena il modello per 40 epoche, usando il dataset di training e validazione, con TensorBoard
    history = model.fit(
        ds_train,
        epochs = 5,
        validation_data = ds_val,
        shuffle = False,
        callbacks = [tensorboard_callback]
    )

    model.evaluate(ds_val, verbose = 2)   # Valuta il modello sul dataset di test
    model.save("analisi_img.keras")   # Salva il modello addestrato

    # Estrae le metriche di loss e accuracy dal training e dalla validazione
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    training_accuracy = history.history['sparse_categorical_accuracy']
    validation_accuracy = history.history['val_sparse_categorical_accuracy']

    # Grafico della loss (errore) durante il training e la validazione
    plt.subplot(1,2,1)
    plt.plot(training_loss, label="Training loss")
    plt.plot(validation_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)

    # Grafico dell'accuracy (precisione) durante il training e la validazione
    plt.subplot(1,2,2)
    plt.plot(training_accuracy, label="Training accuracy")
    plt.plot(validation_accuracy, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()   # Mostra