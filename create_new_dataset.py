import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import os
from scipy import signal

# Charger le modèle entraîné
model = tf.keras.models.load_model('trained_model2.h5')

# Chemin vers le répertoire contenant les fichiers HDF5
# directory_path = '/home/ubuntu/osec/dataset/dataHDF5/templates'
directory_path = 'C:/tmp/OSEC/dh5/transfer_8631648_files_b4ede81a/dataHDF5/templates'

# Utilisation de glob pour lister tous les fichiers .h5 dans le répertoire
file_paths = glob.glob(os.path.join(directory_path, "*.h5"))

# Dossier où enregistrer les résultats
output_directory = 'results'

# Créer le répertoire de sortie s'il n'existe pas
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Parcourir tous les fichiers HDF5
for file_path in file_paths:
    print(f"Traitement du fichier: {file_path}")

    # Ouvrir le fichier HDF5 en mode lecture
    with h5py.File(file_path, 'r') as hdf:
        print("Structure du fichier HDF5:")

        # Dictionnaire pour enregistrer les résultats
        result_dict = {}

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset '{name}' avec forme {obj.shape} et type {obj.dtype}")

                xpoints = np.arange(600)  # Crée un tableau de points x

                ypoints = np.concatenate((obj[:50], obj[:550]))  # Pour recentrer l'onde

                # # sos = signal.butter(50, 10, 'hp', fs=1000, output='sos')
                # # ypoints = signal.sosfilt(sos, ypoints)

                ybis = ypoints
                # Reshape pour avoir la forme (600, 1)
                ypoints = ypoints.reshape(-1, 600, 1)

                # Prédiction du modèle sur les données
                predictions = model.predict(ypoints)

                # Obtenir la classe prédite (0 = up, 1 = down, 2 = unknown)
                predicted_labels = np.argmax(predictions, axis=1)

                # Filtrer les segments où le label est 0 ou 1 ou 2 (up ou down ou unk) à selectionner

                filter = [0,1,2]
                valid_indices = np.where(np.isin(predicted_labels, filter))[0]
                filtered_data = ybis[valid_indices]
                filtered_predictions = predicted_labels[valid_indices]

                # Stocker les résultats filtrés
                if len(filtered_data) > 0:  # Si des segments valides existent
                    result_dict[name] = {
                        'data': filtered_data,  # Données filtrées (up ou down ou  unk)
                        'predictions': filtered_predictions  # Prédictions (0 ou 1 ou 2)
                    }

                # Afficher les résultats filtrés pour chaque segment
                for i, label in enumerate(filtered_predictions):
                    if label == 0:
                        label_str = "up"
                    elif label == 1:
                        label_str = "down"
                    else:
                        label_str = "unk"

                    print(f"Segment {i}: Polarité prédite - {label_str}")

                    plt.title(
                        label_str,
                        loc = "center"
                    )
                    plt.plot(ybis)
                    plt.show()

            elif isinstance(obj, h5py.Group):
                print(f"Groupe '{name}'")

        # Parcours du fichier HDF5 pour afficher la structure
        hdf.visititems(print_structure)

        # Créer un nouveau fichier HDF5 pour enregistrer les résultats filtrés
        output_file_path = os.path.join(output_directory, os.path.basename(file_path))
        with h5py.File(output_file_path, 'w') as output_hdf:
            # Enregistrer les données et les prédictions dans le nouveau fichier, seulement pour les labels filtrés
            for dataset_name, results in result_dict.items():
                # Créer un groupe pour chaque dataset
                group = output_hdf.create_group(dataset_name)

                # Ajouter les données et les prédictions dans ce groupe
                group.create_dataset('data', data=results['data'])
                group.create_dataset('predictions', data=results['predictions'])

        print(f"Les résultats ont été enregistrés dans {output_file_path}")
