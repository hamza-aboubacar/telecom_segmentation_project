# -*- coding: utf-8 -*-
"""
Application Web Flask pour la Segmentation Client Télécom.

Cette application permet de :
1. Afficher un aperçu des segments de clients identifiés.
2. Prédire le segment d'un nouveau client via un formulaire.
"""

from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
import logging

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'votre_cle_secrete_pour_la_segmentation_telecom')

# --- Chargement du modèle et des profils de clusters ---
try:
    model_path = os.path.join(os.path.dirname(__file__), 'telecom_segmentation_model.pkl')
    profiles_path = os.path.join(os.path.dirname(__file__), 'telecom_cluster_profiles.pkl')
    columns_path = os.path.join(os.path.dirname(__file__), 'telecom_expected_input_columns.pkl')

    segmentation_model = joblib.load(model_path)
    cluster_profiles = joblib.load(profiles_path)
    expected_input_columns = joblib.load(columns_path)

    logging.info("Modèle de segmentation, profils de clusters et colonnes attendues chargés avec succès.")

    # Extraire les colonnes numériques et catégorielles originales du préprocesseur
    numeric_features_original = []
    categorical_features_original = []
    for transformer_name, transformer, features_list in segmentation_model.named_steps['preprocessor'].transformers_:
        if transformer_name == 'num':
            numeric_features_original.extend(features_list)
        elif transformer_name == 'cat':
            categorical_features_original.extend(features_list)
    
    logging.info(f"Colonnes numériques originales: {numeric_features_original}")
    logging.info(f"Colonnes catégorielles originales: {categorical_features_original}")

except FileNotFoundError as e:
    logging.error(f"Erreur : Un ou plusieurs fichiers nécessaires sont introuvables. {e}")
    logging.error("Veuillez d'abord exécuter le script 'segmentation_model_creation.py'.")
    exit()
except Exception as e:
    logging.error(f"Erreur lors du chargement des ressources du modèle : {e}", exc_info=True)
    exit()

# --- Routes de l'application ---

@app.route('/')
def index():
    """
    Page d'accueil avec navigation vers la prédiction individuelle et l'aperçu des segments.
    """
    return render_template('telecom_index.html')

@app.route('/segments_overview')
def segments_overview():
    """
    Affiche un tableau récapitulatif des profils de chaque segment.
    """
    # Convertir les profils en un format plus facile à afficher dans HTML
    profiles_df = pd.DataFrame.from_dict(cluster_profiles, orient='index')
    profiles_html = profiles_df.to_html(classes='table-auto w-full text-left whitespace-no-wrap', float_format='%.2f')
    
    return render_template('telecom_segments_overview.html', profiles_table=profiles_html)

@app.route('/segment_predict', methods=['GET', 'POST'])
def segment_predict():
    """
    Permet de saisir les données d'un nouveau client et de prédire son segment.
    """
    predicted_segment_name = None
    predicted_segment_profile = None
    error_message = None

    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges'])
            }

            input_df = pd.DataFrame([data])
            
            # Assurer l'alignement des colonnes et la gestion des types/NaNs
            df_processed = input_df.copy()
            for col in expected_input_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan # Ajouter les colonnes manquantes
            df_processed = df_processed[expected_input_columns] # Réordonner et supprimer les extras

            # Gérer les types de données et les NaN pour la prédiction
            for col in expected_input_columns:
                if col in numeric_features_original:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if df_processed[col].isnull().any():
                        # Pour la prédiction individuelle, on peut utiliser une valeur par défaut globale
                        # ou la médiane du jeu d'entraînement si elle était stockée.
                        # Pour cet exemple, nous utilisons 0 ou la médiane du lot (qui est ici une seule ligne).
                        df_processed[col].fillna(0, inplace=True) # Remplir avec 0 pour une seule ligne
                elif col in categorical_features_original:
                    df_processed[col] = df_processed[col].astype(str).replace('nan', 'Unknown')
            
            # Prédire le cluster
            predicted_cluster_label = segmentation_model.named_steps['kmeans'].predict(
                segmentation_model.named_steps['preprocessor'].transform(df_processed)
            )[0]
            
            predicted_segment_name = f'Cluster {predicted_cluster_label}'
            predicted_segment_profile = cluster_profiles.get(predicted_segment_name, {})
            logging.info(f"Nouveau client prédit dans le segment: {predicted_segment_name}")

        except ValueError as ve:
            error_message = f"Erreur de saisie : {ve}. Veuillez vérifier les valeurs numériques."
            logging.error(error_message, exc_info=True)
        except Exception as e:
            error_message = f"Une erreur inattendue est survenue lors de la prédiction : {e}"
            logging.error(error_message, exc_info=True)

    # Préparer les options pour les listes déroulantes du formulaire
    form_options = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No internet service', 'No', 'Yes'],
        'OnlineBackup': ['No internet service', 'No', 'Yes'],
        'DeviceProtection': ['No internet service', 'No', 'Yes'],
        'TechSupport': ['No internet service', 'No', 'Yes'],
        'StreamingTV': ['No internet service', 'No', 'Yes'],
        'StreamingMovies': ['No internet service', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    return render_template('telecom_segment_predict.html', 
                           predicted_segment_name=predicted_segment_name,
                           predicted_segment_profile=predicted_segment_profile,
                           error_message=error_message,
                           form_options=form_options)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

