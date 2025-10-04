# Données

La classe NHLDataLoader est utilisée pour récupérer les données et la classes StorageData pour les stocker. 

Deux types de données sont enregistrés : le interm (stockage temporaire) et le final (sauvegarde finale).

Pendant la récupération des données, le cache est utilisé pour les stocker temporairement. Cela permet d’exécuter le processus de collecte en plusieurs étapes sans devoir télécharger plusieurs fois les mêmes informations. Le cache est enregistré dans le dossier ift6758/data/storage/cache. Ce chemin peut être modifié à l’aide de la variable d’environnement CACHE_PATH.

Une fois les données d’une saison entièrement récupérées, elles sont enregistrées dans le dump. À la fin du processus de collecte, il est recommandé de vider le cache pour libérer de l’espace. Le dump est enregistré dans le dossier ift6758/data/storage/dump. Ce chemin peut être modifié à l’aide de la variable d’environnement DUMP_PATH.




# Exemple d'utilisation
