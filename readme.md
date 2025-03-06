# Github de développement du PER 10 : Identification de concepts dans les features apprises par un classificateur d'images

## Explication

Le dossier `archive` à la racine contient l'ensemble des versions des fichiers avec notre cheminement et raisonnement.  
Le dossier `src` contient les fichiers dans leurs versions finales utilisés pour obtenir les résultats présentés dans le rapport final.  
Le dossier `src/data` contient l'ensemble des données utilisées, notamment les images qui seront utilisées pour le traitement. Certains dossiers sont vides, mais ils contenaient des fichiers de données en `.npz`, supprimés en raison de leur forte taille (~800Mo chacun). Normalement, les fichiers Jupyter Notebook permettent de les recréer.

## Disclaimer

Les fichiers ne sont pas documentés et sont dans une version plus ou moins brouillonne. Nous avons préféré négliger cette partie afin de passer plus de temps sur la réflexion et l'analyse du projet.

## Signification des noms de fichiers et dossiers (uniquement pour l'archive) :

### Fichiers principaux

- `test_crpv2.ipynb` :
    - Fichier initial pour comprendre le fonctionnement de `zennit-crp` et explorer plusieurs pistes, notamment l'importance des features.

- `app.py` :
    - Lancement avec `❯ streamlit run app.py --server.runOnSave false`
    - Démarre un serveur Streamlit pour extraire les images les plus pertinentes d'un concept à partir d'un dataset spécifique et d'une classe spécifique.

- `testfinetuning2.ipynb` :
    - Teste l'évolution de l'importance des features sur une image unique de `CIFAR-10` avec un modèle `VGG-16` pré-entraîné.

- `testfinetuning2.2.ipynb` :
    - Même principe que `testfinetuning2.ipynb`, mais avec un modèle entraîné depuis zéro.

- `HeatMap_all_features_colab.ipynb` :
    - Exécuté sous Google Colab, génère les heatmaps d'une image selon les couches et features.
    - Calcule le score d'importance de chaque feature pour chaque couche.

### Analyse par clustering

- `cluster_v1.ipynb` :
    - Fichier initial explorant différentes méthodes de clustering pour comprendre la disposition des données et identifier d'éventuels clusters.

- `cluster_v6.ipynb` :
    - Génère une analyse pour une image avec différents algorithmes de clustering et divers nombres de clusters pour identifier la meilleure approche.

- `cluster_v6_visu.ipynb` :
    - Visualise les différents clusters sur les images pour permettre leur annotation.

- `cluster_v6_analysis_v3.ipynb` :
    - Regroupe les clusters de différentes images et assigne chaque feature au cluster le plus fréquent.

- `cluster_v6_visu_after_analysis_v3.ipynb` :
    - Visualisation des résultats finaux pour évaluer la pertinence des clusters obtenus.

- `cluster_v7.ipynb` :
    - Dernière partie du projet : automatisation du double clustering.

### Superclasse ImageNet

- `Superclassing_ImageNet2.2.ipynb` et `Superclassing_ImageNet3.ipynb` :
    - Génération des superclasses avec deux approches différentes.
    - Ces fichiers peuvent ne pas fonctionner à cause de modifications apportées à la librairie utilisée.

### Fichiers supplémentaires

- `requirements.txt` :
    - Liste des librairies requises (sauf peut-être pour `app.py`, où certaines dépendances pourraient manquer).

Dans le dossier `src`, on retrouve aussi :
- `VGG16_ImageNet`
- `zennit-crp` (renommé `zennitcrp`)
Ces dossiers proviennent de tutoriels et sont utilisés par plusieurs fichiers.

## Auteurs et développeurs
- **Arrigoni Guillaume**
- **Juillet Timothée**

## Encadrant
- **Précioso Frédéric**
