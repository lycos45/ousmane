import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Charger les données
data = pd.read_csv("C:/Users/ousma/Documents/Gomycode/Financial_inclusion_dataset.csv")
print("Quelques informatons de notre base")
print(data.info())

print("Voyons combien de donnees manquantes avons-nous ?")
print(data.isnull().sum())

# Créer un dictionnaire pour stocker les encodeurs pour chaque colonne catégorielle
label_encoders = {}

# Encodage des variables catégorielles
columns_encode = ['country', 'uniqueid', 'bank_account', 'location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']
for col in columns_encode:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_encoders[col] = encoder

# Diviser les données en fonctionnalités (X) et cible (y)
x = data[['job_type', 'education_level', 'marital_status', 'age_of_respondent', 'household_size', 'gender_of_respondent']]
y = data['bank_account']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Adapter le modèle aux données d'entraînement
model.fit(X_train, y_train)

# Fonction pour inverser l'encodage
def inverse_encode(column, value):
    return label_encoders[column].inverse_transform([value])[0]

# Interface utilisateur Streamlit
st.title("Prédiction d'Inclusion Financière")

# Formulaire pour les fonctionnalités
job_type = st.selectbox("Type d'emploi", options=label_encoders['job_type'].classes_)
education_level = st.selectbox("Niveau d'éducation", options=label_encoders['education_level'].classes_)
marital_status_options = label_encoders['marital_status'].classes_
marital_status = st.selectbox("Statut matrimonial", options=marital_status_options)
gender_of_respondent = st.radio("Genre du répondant", options=label_encoders['gender_of_respondent'].classes_)
age_of_respondent = st.slider("Âge du répondant", 16, 100)
household_size = st.slider("Taille du ménage", 1, 21)

# Prédiction
if st.button("Prédire l'Inclusion Financière"):
    # Utiliser le modèle pour faire une prédiction avec les valeurs d'entrée
    input_data = [label_encoders['job_type'].transform([job_type])[0],
                  label_encoders['education_level'].transform([education_level])[0],
                  label_encoders['marital_status'].transform([marital_status])[0],
                  age_of_respondent, household_size,
                  label_encoders['gender_of_respondent'].transform([gender_of_respondent])[0]]
    prediction = model.predict([input_data])[0]

    # Convertir la valeur prédite en libellé
    prediction_label = 'Oui' if prediction == 1 else 'Non'

    st.write("Prédiction d'Inclusion Financière:", prediction_label)

