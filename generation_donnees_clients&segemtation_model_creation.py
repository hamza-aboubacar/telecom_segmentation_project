# -*- coding: utf-8 -*-
"""
Script pour générer un jeu de données synthétique de clients télécoms pour la segmentation.

Ce script crée un fichier CSV avec des caractéristiques typiques de clients télécoms,
adaptées à un problème de segmentation.
"""

import pandas as pd
import numpy as np
import random
import logging

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_telecom_segmentation_data(num_clients=5000):
    """
    Génère un DataFrame de données de clients télécoms et l'enregistre en CSV.

    Args:
        num_clients (int): Le nombre de lignes (clients) à générer.
    """
    logging.info(f"Début de la génération de données pour {num_clients} clients...")

    # Caractéristiques démographiques et de contrat
    gender = random.choices(['Male', 'Female'], k=num_clients)
    senior_citizen = random.choices([0, 1], weights=[0.8, 0.2], k=num_clients)
    partner = random.choices(['Yes', 'No'], weights=[0.5, 0.5], k=num_clients)
    dependents = random.choices(['Yes', 'No'], weights=[0.3, 0.7], k=num_clients)
    contract = random.choices(['Month-to-month', 'One year', 'Two year'], weights=[0.5, 0.3, 0.2], k=num_clients)
    paperless_billing = random.choices(['Yes', 'No'], weights=[0.6, 0.4], k=num_clients)
    payment_method = random.choices(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], weights=[0.3, 0.2, 0.25, 0.25], k=num_clients)

    # Caractéristiques de service
    phone_service = random.choices(['Yes', 'No'], weights=[0.9, 0.1], k=num_clients)
    multiple_lines = [random.choice(['No', 'Yes', 'No phone service']) if ps == 'Yes' else 'No phone service' for ps in phone_service]
    internet_service = random.choices(['DSL', 'Fiber optic', 'No'], weights=[0.35, 0.45, 0.2], k=num_clients)
    
    # Services additionnels (dépendent du service internet)
    online_security = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]
    online_backup = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]
    device_protection = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]
    tech_support = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]
    streaming_tv = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]
    streaming_movies = [random.choice(['Yes', 'No']) if i_s != 'No' else 'No internet service' for i_s in internet_service]

    # Caractéristiques numériques
    tenure_months = np.random.randint(1, 73, num_clients) # 1 à 72 mois
    monthly_charges = np.random.uniform(20.0, 120.0, num_clients)
    
    # Simuler TotalCharges basé sur MonthlyCharges et tenure
    total_charges = []
    for i in range(num_clients):
        if tenure_months[i] == 0: # Pour les nouveaux clients sans ancienneté
            total_charges.append(monthly_charges[i])
        else:
            total_charges.append(monthly_charges[i] * tenure_months[i] * np.random.uniform(0.9, 1.1)) # Ajout de bruit
    total_charges = np.array(total_charges).round(2)

    # Créer le DataFrame
    data = {
        'customerID': [f'C{i:04d}' for i in range(1, num_clients + 1)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure_months,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges.round(2),
        'TotalCharges': total_charges
    }
    df = pd.DataFrame(data)

    # Introduire quelques valeurs manquantes pour le test
    # (Par exemple, dans TotalCharges pour simuler de nouveaux clients ou erreurs)
    num_nulls_total_charges = int(num_clients * 0.01) # 1% de nulls
    null_indices = np.random.choice(df.index, num_nulls_total_charges, replace=False)
    df.loc[null_indices, 'TotalCharges'] = np.nan
    logging.info(f"Introduit {num_nulls_total_charges} valeurs nulles dans 'TotalCharges'.")

    # Sauvegarder le DataFrame dans un fichier CSV
    file_name = 'telecom_customer_data.csv'
    df.to_csv(file_name, index=False)
    logging.info(f"Fichier '{file_name}' généré avec succès. Il contient {len(df)} lignes.")
    logging.info("Utilisez ce fichier pour entraîner et tester votre application.")

if __name__ == '__main__':
    generate_telecom_segmentation_data()




# -*- coding: utf-8 -*-
"""
Script pour l'entraînement du modèle de segmentation client télécom.

Ce script charge les données générées, effectue le prétraitement,
applique l'algorithme de clustering K-Means, analyse les profils de clusters
et sauvegarde le modèle et les profils.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import os

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_segmentation_model():
    """
    Charge les données, entraîne un modèle de segmentation (K-Means) et le sauvegarde.
    Analyse également les profils des clusters.
    """
    logging.info("Démarrage de la création du modèle de segmentation...")

    data_file = 'telecom_customer_data.csv'
    if not os.path.exists(data_file):
        logging.error(f"Erreur : Le fichier de données '{data_file}' est introuvable.")
        logging.error("Veuillez d'abord exécuter le script 'generate_telecom_segmentation_data.py'.")
        return

    try:
        df = pd.read_csv(data_file)
        logging.info("Jeu de données chargé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        return

    # --- Nettoyage et préparation des données ---
    df.drop('customerID', axis=1, inplace=True, errors='ignore')

    # Gérer les valeurs manquantes dans TotalCharges (introduites lors de la génération)
    # Remplir avec la médiane avant le prétraitement
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    logging.info(f"Valeurs manquantes dans 'TotalCharges' remplies avec la médiane : {median_total_charges}")

    # Définir les colonnes numériques et catégorielles
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    logging.info(f"Colonnes numériques pour le prétraitement: {numeric_features}")
    logging.info(f"Colonnes catégorielles pour le prétraitement: {categorical_features}")

    # Créer un préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Conserver les colonnes non spécifiées (ici, il ne devrait pas y en avoir)
    )

    # Créer un pipeline avec le préprocesseur et K-Means
    # Nous allons arbitrairement choisir 4 clusters pour cet exemple.
    # Dans un vrai projet, vous utiliseriez la méthode du coude ou le score de silhouette.
    n_clusters = 4
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10)) # n_init pour éviter les avertissements
    ])

    logging.info(f"Entraînement du modèle K-Means avec {n_clusters} clusters...")
    model_pipeline.fit(df) # Entraîner sur le DataFrame complet pour la segmentation

    # Obtenir les labels de cluster pour chaque client
    df['Cluster'] = model_pipeline.named_steps['kmeans'].labels_
    logging.info("Clustering terminé. Labels de cluster ajoutés au DataFrame.")

    # --- Analyse et Interprétation des Clusters ---
    cluster_profiles = {}
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i].drop('Cluster', axis=1)
        profile = {}
        
        # Pour les caractéristiques numériques, calculer la moyenne
        for col in numeric_features:
            profile[col] = cluster_data[col].mean().round(2)
        
        # Pour les caractéristiques catégorielles, calculer le mode (la catégorie la plus fréquente)
        for col in categorical_features:
            profile[col] = cluster_data[col].mode()[0] if not cluster_data[col].empty else 'N/A'
        
        cluster_profiles[f'Cluster {i}'] = profile
    
    logging.info("Profils de clusters générés.")
    
    # Sauvegarder le pipeline complet (préprocesseur + K-Means)
    joblib.dump(model_pipeline, 'telecom_segmentation_model.pkl')
    logging.info("Modèle de segmentation sauvegardé sous 'telecom_segmentation_model.pkl'.")

    # Sauvegarder les profils des clusters
    joblib.dump(cluster_profiles, 'telecom_cluster_profiles.pkl')
    logging.info("Profils de clusters sauvegardés sous 'telecom_cluster_profiles.pkl'.")

    # Sauvegarder les noms des colonnes d'entrée attendues par le préprocesseur
    # C'est crucial pour l'application Flask
    expected_input_columns = model_pipeline.named_steps['preprocessor'].feature_names_in_
    joblib.dump(expected_input_columns, 'telecom_expected_input_columns.pkl')
    logging.info("Noms des colonnes d'entrée attendues sauvegardés sous 'telecom_expected_input_columns.pkl'.")

    logging.info("Processus de création du modèle de segmentation terminé.")

if __name__ == '__main__':
    create_segmentation_model()
