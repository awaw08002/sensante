"""
SenSante - Exploration du dataset patients_dakar.csv.csv
Lab 1 : Git, Python et Structure Projet
"""
import pandas as pd

# Charger les données
df = pd.read_csv("data/patients_dakar.csv.csv")

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

print("\n--- Répartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

print("\n--- Répartition par région (top 5) ---")
for region, count in df["region"].value_counts().head(5).items():
    print(f"  {region:15s} : {count:3d} patients")

print("\n--- Température moyenne par diagnostic ---")
for diag, temp in df.groupby("diagnostic")["temperature"].mean().items():
    print(f"  {diag:12s} : {temp:.1f}°C")

print("\n" + "=" * 50)
print("Exploration terminée !")
print("Prochain lab : entraîner un modèle ML")
print("=" * 50)


print("\n--- Patients par sexe et diagnostic ---")
print(df.groupby(["sexe", "diagnostic"]).size())