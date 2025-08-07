# Projet : Segmentation Client Télécom et Optimisation des Offres
## 📝 Description du Projet
Ce projet complet vise à aider les entreprises de télécommunications à mieux comprendre leur clientèle en les divisant en segments distincts. En identifiant des groupes de clients ayant des comportements et des caractéristiques similaires, les opérateurs peuvent personnaliser leurs stratégies marketing, optimiser leurs offres de services et améliorer la satisfaction client.

Le projet inclut la génération de données synthétiques, l'entraînement d'un modèle de clustering K-Means, et une application web Flask interactive pour visualiser les segments et prédire le segment d'un nouveau client.

## 🎯 Problème Métier Addréssé
Dans un marché concurrentiel, comprendre les besoins spécifiques des différents types de clients est crucial. Ce projet répond à la question suivante : "Comment identifier des groupes homogènes de clients pour leur proposer des offres plus pertinentes, améliorer leur fidélité et optimiser les efforts marketing ?"

## ✨ Fonctionnalités
Génération de Données Synthétiques : Crée un jeu de données réaliste de clients télécoms pour simuler un scénario réel.

Modèle de Segmentation (Clustering K-Means) : Entraîne un modèle pour regrouper les clients en segments distincts basés sur leurs caractéristiques.

Analyse des Profils de Segments : Génère des descriptions détaillées pour chaque segment, mettant en lumière leurs caractéristiques uniques.

Application Web Flask :

Aperçu des Segments : Affiche un tableau récapitulatif des profils de tous les segments identifiés.

Prédiction Individuelle de Segment : Permet de saisir les informations d'un nouveau client via un formulaire et de prédire à quel segment il appartient.

Robustesse des Données : Le système est conçu pour gérer les valeurs manquantes dans les données d'entrée.

## 💻 Technologies et Dépendances
Python 3.x

Framework Web : Flask

Data Science : pandas, numpy, scikit-learn

Sérialisation : joblib

Serveur WSGI (pour le déploiement) : gunicorn

Frontend : HTML, CSS (avec Tailwind CSS via CDN)

Versionnement : Git, GitHub

## 📁 Structure du Projet
telecom_segmentation_project/
├── app.py                         # Application web Flask
├── generate_telecom_segmentation_data.py # Script de génération de données
├── segmentation_model_creation.py # Script d'entraînement du modèle de segmentation
├── telecom_customer_data.csv      # Jeu de données client généré (sera créé)
├── telecom_segmentation_model.pkl # Modèle K-Means sauvegardé (sera créé)
├── telecom_cluster_profiles.pkl   # Profils des segments sauvegardés (sera créé)
├── telecom_expected_input_columns.pkl # Noms des colonnes d'entrée attendues (sera créé)
└── templates/                     # Dossier des templates HTML
    ├── telecom_index.html
    ├── telecom_segment_predict.html
    └── telecom_segments_overview.html

## 🚀 Étapes de Réalisation du Projet (De A à Z)
Suivez ces étapes pour mettre en place et exécuter le projet.

1. **Génération des Données**
   
Ce script crée un jeu de données synthétique de clients télécoms, simulant des caractéristiques variées pour la segmentation.

Fichier : generate_telecom_segmentation_data.py

Objectif : Créer le fichier telecom_customer_data.csv.

Comment l'exécuter :

python generate_telecom_segmentation_data.py

Output attendu : Un fichier telecom_customer_data.csv sera généré à la racine de votre projet, contenant 5000 lignes de données clients fictives.

2. **Création du Modèle de Segmentation**

Ce script utilise les données générées pour entraîner un modèle de clustering K-Means et analyser les caractéristiques de chaque segment.

Fichier : segmentation_model_creation.py

Objectif :

Charger telecom_customer_data.csv.

Prétraiter les données (gestion des valeurs manquantes dans TotalCharges, encodage des catégories, standardisation des numériques).

Appliquer K-Means pour segmenter les clients (par défaut, 4 clusters).

Analyser les profils de chaque cluster (moyennes pour les numériques, modes pour les catégorielles).

Sauvegarder le pipeline complet du modèle (telecom_segmentation_model.pkl), les profils des clusters (telecom_cluster_profiles.pkl), et les noms des colonnes d'entrée attendues (telecom_expected_input_columns.pkl).

Comment l'exécuter :

python segmentation_model_creation.py

Output attendu : Trois fichiers .pkl seront créés à la racine de votre projet, essentiels pour l'application Flask.

3. **Lancement de l'Application Web Flask**
   
Cette application web permet d'interagir avec le modèle de segmentation.

Fichier : app.py

Objectif :

Charger le modèle et les profils de segments.

Servir les pages web pour l'aperçu des segments et la prédiction individuelle.

Traiter les entrées du formulaire et utiliser le modèle pour prédire le segment d'un client.

Comment l'exécuter :

python app.py

Output attendu : L'application démarrera et sera accessible via votre navigateur web.

# 📊 Explication de l'Output (Résultats)
L'application Flask propose deux types d'outputs principaux :

1. **Aperçu des Segments**
Page : /segments_overview (accessible depuis la page d'accueil)

Description : Cette page affiche un tableau récapitulatif des caractéristiques moyennes (pour les numériques) et des catégories les plus fréquentes (pour les catégorielles) de chaque segment client identifié par le modèle K-Means.

Utilité : Permet aux analystes et aux équipes marketing de comprendre les "personas" de chaque segment (ex: "Segment 0 : Jeunes clients fibre optique à forte dépense mensuelle", "Segment 1 : Seniors fidèles avec contrat de 2 ans"). Ces informations sont cruciales pour adapter les offres, les messages publicitaires et les stratégies de rétention.

2. **Prédiction du Segment d'un Client**
Page : /segment_predict (accessible depuis la page d'accueil)

Description : L'utilisateur remplit un formulaire avec les caractéristiques d'un nouveau client ou d'un client existant. L'application utilise le modèle de segmentation pour déterminer à quel segment ce client est le plus susceptible d'appartenir. Le profil du segment prédit est ensuite affiché.

Utilité : Permet une action immédiate et personnalisée. Par exemple, si un nouveau client est prédit dans le "Segment des clients à forte valeur", l'équipe commerciale peut lui proposer une offre premium dès le départ. Si un client existant est prédit dans un segment à risque, des actions de rétention spécifiques à ce segment peuvent être déclenchées.

# 🚀 Installation et Démarrage Local
Pour faire tourner ce projet sur votre machine :

Cloner le dépôt :

git clone https://github.com/hamza-aboubacar/telecom_segmentation_project.git
cd telecom_segmentation_project

Créer et activer un environnement virtuel (recommandé) :

python -m venv venv_telecom_segmentation
# Sur macOS/Linux:
source venv_telecom_segmentation/bin/activate
# Sur Windows:
venv_telecom_segmentation\Scripts\activate

Installer les dépendances :

pip install pandas numpy scikit-learn Flask joblib gunicorn

Générer les données :

python generate_telecom_segmentation_data.py

Créer le modèle de segmentation :

python segmentation_model_creation.py

Lancer l'application Flask :

python app.py

L'application sera disponible à l'adresse http://127.0.0.1:5000.

## Quelques captures d'images
<img width="897" height="347" alt="Capture d’écran 2025-08-07 114043" src="https://github.com/user-attachments/assets/39f4ed80-0419-495a-bfe5-cd455c7652da" />

<img width="1445" height="757" alt="Capture d’écran 2025-08-07 114003" src="https://github.com/user-attachments/assets/ba1fe81e-37ff-40e0-9f19-f6dc9a384666" />

<img width="1113" height="897" alt="Capture d’écran 2025-08-07 113919" src="https://github.com/user-attachments/assets/740b8fa0-890a-46a1-a423-d72774425539" />






## ✍️ Auteur
Aboubacar Halidou Hamza

[Votre Profil GitHub](https://github.com/hamza-aboubacar/telecom_segmentation_project)


📄 Licence
Ce projet est sous licence MIT.
