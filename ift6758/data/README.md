# Données

La classe _NHLDataLoader_ est utilisée pour récupérer les données et la classes _StorageData_ pour les stocker. 

Deux types de données sont enregistrés : le _interm_ (stockage temporaire) et le _final_ (sauvegarde finale).

Pendant la récupération des données, le _interm_ est utilisé pour les stocker temporairement. Cela permet d’exécuter le processus de collecte en plusieurs étapes sans devoir télécharger plusieurs fois les mêmes informations. Le _interm_ est enregistré dans le dossier ift6758/data/sauvegarde/interm. Ce chemin peut être modifié à l’aide de la variable d’environnement INTERIM_PATH.

Une fois les données d’une saison récupérées, elles sont enregistrées dans le _final_. À la fin du processus de collecte, il est recommandé de vider le cache pour libérer de l’espace. Le _final_ est enregistré dans le dossier ift6758/data/sauvegarde/final. Ce chemin peut être modifié à l’aide de la variable d’environnement FINAL_PATH.


# Exemple d'utilisation
