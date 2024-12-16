import os
import requests
from tqdm import tqdm  # Importation de tqdm pour la barre de progression

# Liste des URLs des fichiers à télécharger
urls = [
    "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5",
    "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5"
]

# Fonction pour télécharger un fichier avec suivi de l'avancement
def download_file(url):
    # Récupérer le nom du fichier à partir de l'URL
    file_name = url.split("/")[-1]

    # Vérifier si le fichier existe déjà dans le répertoire courant
    if os.path.exists(file_name):
        print(f"Le fichier {file_name} existe déjà.")
        return

    # Télécharger le fichier
    print(f"Téléchargement du fichier {file_name}...")
    try:
        # Effectuer la requête GET pour récupérer le fichier
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Obtenir la taille du fichier à télécharger
        total_size = int(response.headers.get('Content-Length', 0))

        # Utiliser tqdm pour afficher une barre de progression
        with open(file_name, 'wb') as f:
            # Télécharger par morceaux de 1024 octets
            for data in tqdm(response.iter_content(chunk_size=1024),
                              total=total_size // 1024,
                              unit='KB',
                              desc=file_name):
                f.write(data)

        print(f"Fichier {file_name} téléchargé avec succès.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement de {file_name}: {e}")

# Télécharger chaque fichier
for url in urls:
    download_file(url)
