import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import h5py
import numpy as np
import os
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import h5py
import datetime
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam

# Désactiver les avertissements inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprime les logs TensorFlow
warnings.filterwarnings('ignore')  # Supprime les warnings Python

# Fichiers de données
train_file_path = 'dataset/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5'
test_file_path = 'dataset/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5'

# Vérification et configuration du GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Utilisation du GPU détectée.")
    except RuntimeError as e:
        print(e)
else:
    print("Aucun GPU détecté. Entraînement sur CPU.")

# Générateur pour charger les données depuis un fichier HDF5
def data_generator(hdf5_file, batch_size=64):
    """
    Générateur pour lire et charger les données par batch depuis un fichier HDF5.
    """
    with h5py.File(hdf5_file, 'r') as hdf:
        X = hdf['X'][:]
        fm = hdf['fm'][:]

        # Vérification des classes présentes dans les données
        unique_classes = np.unique(fm)
        print(f"Classes uniques dans les données: {unique_classes}")

        # Reshaper X de (N, 600) à (N, 600, 1)
        X = X.reshape(-1, 600, 1)
        num_batches = len(X) // batch_size

        # Génération des batches
        while True:
            for i in range(num_batches):
                X_batch = X[i * batch_size:(i + 1) * batch_size]
                fm_batch = fm[i * batch_size:(i + 1) * batch_size]
                yield X_batch, fm_batch

# Charger les données pour déterminer la taille
def get_dataset_size(hdf5_file):
    """Retourne la taille du dataset dans un fichier HDF5."""
    with h5py.File(hdf5_file, 'r') as hdf:
        size = hdf['X'].shape[0]
    print(f"Nombre total d'exemples dans {hdf5_file}: {size}")
    return size

train_size = get_dataset_size(train_file_path)
test_size = get_dataset_size(test_file_path)

# Architecture du modèle CNN
model = Sequential([
    Input(shape=(600, 1)),
    Conv1D(32, kernel_size=7, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(128, activation='tanh'),

    Dropout(0.5),
    Dense(512, activation='relu'),

    Dropout(0.5),
    Dense(512, activation='tanh'),

    Dropout(0.5),
    Dense(3, activation='softmax')
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Afficher un résumé du modèle
model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle
batch_size = 64
epochs = 20

print("\nDémarrage de l'entraînement...")
history = model.fit(
    data_generator(train_file_path, batch_size=batch_size),
    steps_per_epoch=(train_size // batch_size),
    epochs=epochs,
    validation_data=data_generator(test_file_path, batch_size=batch_size),
    validation_steps=(test_size // batch_size),
    callbacks=[early_stopping,tensorboard_callback]
)

# Évaluation sur le dataset de test
print("\nÉvaluation sur le dataset de test...")
loss, accuracy = model.evaluate(
    data_generator(test_file_path, batch_size=batch_size),
    steps=(test_size // batch_size)
)
print(f"\nPrécision sur l'ensemble de test: {accuracy * 100:.2f}%")

# Enregistrement du modèle
model_save_path = 'model/trained_model' + str(epochs) + '.h5'
print(f"\nEnregistrement du modèle dans {model_save_path}...")
model.save(model_save_path)
print("Modèle enregistré avec succès.")