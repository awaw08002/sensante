# notebooks/train_model.py
# Lab 2 : Entrainer et Serialiser un Modele - SenSante
# ESP/UCAD - L2 GLSI

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================
# ETAPE 2 : Charger et preparer les donnees
# ==============================================================

# Etape 2.1 : Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

print("Dataset : " + str(df.shape[0]) + " patients, " + str(df.shape[1]) + " colonnes")
print("\nColonnes : " + str(list(df.columns)))
print("\nDiagnostics :")
print(df['diagnostic'].value_counts())

# Etape 2.2 : Preparer les features et la cible
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print("Features : " + str(X.shape))
print("Cible : " + str(y.shape))


# ==============================================================
# ETAPE 3 : Separer entrainement et test
# ==============================================================

# Etape 3.1 : Separer les donnees
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrainement : " + str(X_train.shape[0]) + " patients")
print("Test : " + str(X_test.shape[0]) + " patients")


# ==============================================================
# ETAPE 4 : Entrainer le modele
# ==============================================================

# Etape 4.1 : Entrainer le modele
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print("Nombre d'arbres : " + str(model.n_estimators))
print("Nombre de features : " + str(model.n_features_in_))
print("Classes : " + str(list(model.classes_)))


# ==============================================================
# ETAPE 5 : Evaluer le modele
# ==============================================================

# Etape 5.1 : Predire sur les donnees de test
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

# Etape 5.2 : Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : " + str(round(accuracy * 100, 2)) + "%")

# Etape 5.3 : Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Etape 5.4 : Visualiser la matrice de confusion
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")


# ==============================================================
# ETAPE 6 : Serialiser le modele
# ==============================================================

# Etape 6.1 : Sauvegarder le modele
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print("Modele sauvegarde : models/model.pkl")
print("Taille : " + str(round(size / 1024, 1)) + " Ko")

# Etape 6.2 : Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")


# ==============================================================
# ETAPE 7 : Tester le modele serialise
# ==============================================================

# Etape 7.1 : Recharger le modele depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print("Modele recharge : " + type(model_loaded).__name__)
print("Classes : " + str(list(model_loaded.classes_)))

# Etape 7.2 : Predire pour un nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Resultat du pre-diagnostic ---")
print("Patient : " + nouveau_patient['sexe'] + ", " + str(nouveau_patient['age']) + " ans")
print("Diagnostic : " + diagnostic)
print("Probabilite : " + str(round(proba_max * 100, 1)) + "%")
print("\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print("  " + classe + " : " + str(round(proba * 100, 1)) + "% " + bar)


# ==============================================================
# EXERCICE 1 : Importance des features
# ==============================================================

print("\n" + "=" * 60)
print("EXERCICE 1 - Importance des features")
print("=" * 60)

importances = model.feature_importances_
print("\nClassement des features par importance :")
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print("  " + name + " : " + str(round(imp, 3)))

plt.figure(figsize=(8, 5))
feat_imp_sorted = sorted(zip(feature_cols, importances), key=lambda x: x[1])
noms = [f[0] for f in feat_imp_sorted]
vals = [f[1] for f in feat_imp_sorted]
plt.barh(noms, vals, color='steelblue')
plt.xlabel('Importance')
plt.title('Importance des features - RandomForest SenSante')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/feature_importance.png")


# ==============================================================
# EXERCICE 2 : Tester avec 3 patients fictifs
# ==============================================================

print("\n" + "=" * 60)
print("EXERCICE 2 - Tester avec 3 patients fictifs")
print("=" * 60)

patients_fictifs = [
    {
        'nom': 'Patient 1 - Jeune sans symptomes (20 ans, M)',
        'age': 20,
        'sexe': 'M',
        'temperature': 37.0,
        'tension_sys': 120,
        'toux': False,
        'fatigue': False,
        'maux_tete': False,
        'region': 'Dakar'
    },
    {
        'nom': 'Patient 2 - Adulte avec forte fievre (40 ans, F)',
        'age': 40,
        'sexe': 'F',
        'temperature': 40.5,
        'tension_sys': 130,
        'toux': True,
        'fatigue': True,
        'maux_tete': True,
        'region': 'Thies'
    },
    {
        'nom': 'Patient 3 - Personne agee avec toux (68 ans, M)',
        'age': 68,
        'sexe': 'M',
        'temperature': 38.2,
        'tension_sys': 145,
        'toux': True,
        'fatigue': True,
        'maux_tete': False,
        'region': 'Saint-Louis'
    }
]

for p in patients_fictifs:
    s_enc = le_sexe_loaded.transform([p['sexe']])[0]
    r_enc = le_region_loaded.transform([p['region']])[0]

    feats = [
        p['age'],
        s_enc,
        p['temperature'],
        p['tension_sys'],
        int(p['toux']),
        int(p['fatigue']),
        int(p['maux_tete']),
        r_enc
    ]

    diag = model_loaded.predict([feats])[0]
    prbs = model_loaded.predict_proba([feats])[0]
    conf = prbs.max()

    print("\n" + p['nom'])
    print("  -> Diagnostic : " + diag + "  (confiance : " + str(round(conf * 100, 1)) + "%)")
    for classe, proba in zip(model_loaded.classes_, prbs):
        bar = '#' * int(proba * 25)
        print("     " + classe + " : " + str(round(proba * 100, 1)) + "% " + bar)


# ==============================================================
# EXERCICE 3 : Reflexion sur l'accuracy
# ==============================================================

print("\n" + "=" * 60)
print("EXERCICE 3 - Reflexion : 89% d'accuracy, est-ce suffisant ?")
print("=" * 60)
print("""
Dans un contexte medical reel, une accuracy de 89% n'est pas
suffisante pour un deploiement autonome. Sur 1000 patients,
110 recevraient un diagnostic errone. Un faux negatif (ex :
diagnostiquer sain un patient atteint de paludisme) peut
entrainer un retard de traitement potentiellement fatal. Un
faux positif (ex : typh pour un patient sain) expose a des
traitements inutiles et dangereux. Ce modele doit rester un
outil d'aide a la decision pour le personnel de sante,
jamais un substitut au medecin.
""")