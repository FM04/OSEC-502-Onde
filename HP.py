# Importation des bibliothèques
# TensorFlow pour construire le modèle, et TensorBoard pour visualiser les hyperparamètres
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler  # Importé mais pas utilisé (peut être supprimé)
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import h5py
import datetime
from tensorboard.plugins.hparams import api as hp

# Configuration du GPU (si disponible)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Autorisation de l'allocation dynamique de mémoire pour éviter un blocage complet de la mémoire GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Mémoire GPU configurée.")
    except RuntimeError as e:
        print(e)

# Fonction pour générer des données à partir d'un fichier HDF5
def data_generator(hdf5_file, batch_size):
    """
    Générateur pour lire et charger les données par batch depuis un fichier HDF5.
    Le fichier est ouvert à chaque appel pour charger les données.

    Arguments :
        hdf5_file (str) : Chemin vers le fichier HDF5.
        batch_size (int) : Taille des batches à charger.

    Retour :
        yield : Lot de données X et Y.
    """
    with h5py.File(hdf5_file, 'r') as hdf:
        num_samples = hdf['X'].shape[0]  # Nombre total d'échantillons

    num_batches = num_samples // batch_size  # Nombre total de batches

    while True:
        for i in range(num_batches):
            with h5py.File(hdf5_file, 'r') as hdf:
                # Charger un batch de données
                X_batch = hdf['X'][i*batch_size:(i+1)*batch_size]
                Y_batch = hdf['fm'][i*batch_size:(i+1)*batch_size]
                X_batch = X_batch.reshape(-1, 600, 1)  # Reshape pour correspondre à l'entrée du modèle
            yield X_batch, Y_batch

# Chargement des données
data_batch_size = 2**18  # Taille très grande (vérifiez la mémoire disponible)
train_file_path = '../dataset/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5'
test_file_path = '../dataset/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5'

# Génération des premiers batches pour entraînement et test
X_batch, Y_batch = next(data_generator(train_file_path, batch_size=data_batch_size))
X_test_batch, Y_test_batch = next(data_generator(test_file_path, batch_size=data_batch_size))

# Définition des hyperparamètres pour l'optimisation
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_CONV_FILTERS = hp.HParam('conv_filters', hp.Discrete([8, 16, 32, 64, 128]))
HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([8, 16, 32, 64]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001]))

# Activation des couches
activation = hp.Discrete(['relu'])
HP_ACTIVATION_CONV1 = hp.HParam('activation_conv1', activation)
HP_ACTIVATION_CONV2 = hp.HParam('activation_conv2', activation)
HP_ACTIVATION_CONV3 = hp.HParam('activation_conv3', activation)
HP_ACTIVATION_DENSE1 = hp.HParam('activation_dense1', activation)
HP_ACTIVATION_DENSE2 = hp.HParam('activation_dense2', activation)
HP_ACTIVATION_DENSE3 = hp.HParam('activation_dense3', activation)

METRIC_ACCURACY = 'accuracy'

# Liste des hyperparamètres
hparams = [
    HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_CONV_FILTERS, HP_KERNEL_SIZE, HP_LEARNING_RATE,
    HP_ACTIVATION_CONV1, HP_ACTIVATION_CONV2, HP_ACTIVATION_CONV3,
    HP_ACTIVATION_DENSE1, HP_ACTIVATION_DENSE2, HP_ACTIVATION_DENSE3
]

# Fonction pour calculer le nombre de combinaisons possibles
def calculate_combinations_from_hparams(hparams, real_interval_resolution=2):
    """
    Calcule le nombre total de combinaisons possibles des hyperparamètres.

    Arguments :
        hparams (list) : Liste des objets hp.HParam définissant les hyperparamètres.
        real_interval_resolution (int) : Résolution pour les intervalles continus (non utilisé ici).

    Retour :
        int : Nombre total de combinaisons possibles.
    """
    from math import prod

    counts = []
    for hparam in hparams:
        domain = hparam.domain
        if isinstance(domain, hp.Discrete):
            counts.append(len(domain.values))
        else:
            raise ValueError(f"Domaine non supporté : {hparam.name}")

    return prod(counts)

# Calcul et affichage des combinaisons possibles
num_combinations = calculate_combinations_from_hparams(hparams)
print(f"Nombre total de combinaisons possibles : {num_combinations}")

# Configuration TensorBoard pour le suivi des hyperparamètres
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=hparams,
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

# Fonction d'entraînement et de test du modèle
def train_test_model(hparams, epoch=1):
    """
    Entraîne et évalue un modèle avec des hyperparamètres donnés.

    Arguments :
        hparams (dict) : Hyperparamètres utilisés pour construire le modèle.
        epoch (int) : Nombre d'époques d'entraîment.

    Retour :
        float : Précision sur le jeu de test.
    """
    optimizer_name = hparams[HP_OPTIMIZER].lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])
    else:
        raise ValueError(f"Optimiseur non supporté : {optimizer_name}")

    # Définition du modèle
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(600, 1), name='input_layer'),

        # Couches convolutives
        tf.keras.layers.Conv1D(filters=hparams[HP_CONV_FILTERS],
                               kernel_size=hparams[HP_KERNEL_SIZE],
                               activation=hparams[HP_ACTIVATION_CONV1],
                               padding='same',
                               name='conv1'),
        tf.keras.layers.MaxPooling1D(pool_size=2, name='pool1'),

        tf.keras.layers.Conv1D(filters=hparams[HP_CONV_FILTERS],
                               kernel_size=hparams[HP_KERNEL_SIZE],
                               activation=hparams[HP_ACTIVATION_CONV2],
                               padding='same',
                               name='conv2'),
        tf.keras.layers.MaxPooling1D(pool_size=2, name='pool2'),

        tf.keras.layers.Conv1D(filters=hparams[HP_CONV_FILTERS],
                               kernel_size=hparams[HP_KERNEL_SIZE],
                               activation=hparams[HP_ACTIVATION_CONV3],
                               padding='same',
                               name='conv3'),
        tf.keras.layers.MaxPooling1D(pool_size=2, name='pool3'),

        tf.keras.layers.Flatten(name='flatten'),

        # Couches denses
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION_DENSE1], name='dense1'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT], name='dropout1'),

        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION_DENSE2], name='dense2'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT], name='dropout2'),

        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION_DENSE3], name='dense3'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT], name='dropout3'),

        tf.keras.layers.Dense(3, activation='softmax', name='output_layer')
    ])

    # Compilation du modèle
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraîne le modèle
    model.fit(X_batch, Y_batch, epochs=epoch, verbose=0)

    # Évaluation
    _, accuracy = model.evaluate(X_test_batch, Y_test_batch, verbose=0)
    return accuracy

# Lancer une expérience avec les hyperparamètres définis
def run(run_dir, hparams):
    """
    Lance une expérience avec TensorBoard.

    Arguments :
        run_dir (str) : Dossier pour les logs TensorBoard.
        hparams (dict) : Hyperparamètres à utiliser.
    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # Log des hyperparamètres
        accuracy = train_test_model(hparams, epoch=5)  # Entraînement et évaluation
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)  # Log de la précision

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in HP_DROPOUT.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                for activation_dense1 in HP_ACTIVATION_DENSE1.domain.values:
                    for activation_dense2 in HP_ACTIVATION_DENSE2.domain.values:
                        for activation_dense3 in HP_ACTIVATION_DENSE3.domain.values:
                            for activation_conv1 in HP_ACTIVATION_CONV1.domain.values:
                                for activation_conv2 in HP_ACTIVATION_CONV2.domain.values:
                                    for activation_conv3 in HP_ACTIVATION_CONV3.domain.values:
                                        for kernel_size in  HP_KERNEL_SIZE.domain.values:
                                            for conv_filters in HP_CONV_FILTERS.domain.values:
                                                # Définir les hyperparamètres pour cette itération
                                                hparams = {
                                                    HP_NUM_UNITS: num_units,
                                                    HP_DROPOUT: dropout_rate,
                                                    HP_OPTIMIZER: optimizer,
                                                    HP_ACTIVATION_DENSE1: activation_dense1,
                                                    HP_ACTIVATION_DENSE2: activation_dense2,
                                                    HP_ACTIVATION_DENSE3: activation_dense3,
                                                    HP_LEARNING_RATE: learning_rate,
                                                    HP_CONV_FILTERS: conv_filters,
                                                    HP_ACTIVATION_CONV1: activation_conv1,
                                                    HP_ACTIVATION_CONV2: activation_conv2,
                                                    HP_ACTIVATION_CONV3: activation_conv3,
                                                    HP_KERNEL_SIZE: kernel_size
                                                }

                                                # Lancer une exécution avec ce jeu d'hyperparamètres
                                                run_name = f"run-{session_num}"
                                                print(f"--- Starting trial: {run_name}")
                                                print({h.name: hparams[h] for h in hparams})

                                                # Appeler la fonction d'entraînement avec les hyperparamètres
                                                run(f'logs/hparam_tuning/{run_name}', hparams)

                                                session_num += 1