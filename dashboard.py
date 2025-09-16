import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# ===============================
# CONFIGURATION DU DASHBOARD
# ===============================
st.set_page_config(page_title="Dashboard POC - Comparaison Baseline vs CLIP", layout="wide")

st.title("Proof of Concept : Comparaison Baseline vs Modèles CLIP")
st.markdown("""
Ce dashboard présente une comparaison entre le modèle **Baseline (VGG16)** et plusieurs variantes de l’algorithme **CLIP**.  
L’objectif est de mettre en évidence les gains de performance obtenus grâce à CLIP, ainsi que les différences liées 
aux choix de prétraitement (tokenisation, troncature, etc.).
""")

# ===============================
# SECTION 1 : Présentation des modèles
# ===============================
st.header("Présentation des modèles")

with st.expander("Détails techniques des modèles"):
    st.subheader("Baseline")
    st.write("- Algorithme : VGG16 avec data augmentation")
    st.write("- Particularités : modèle CNN classique, utilisé comme référence de comparaison")

    st.subheader("Modèles CLIP")
    st.write("- Algorithme : CLIP (Contrastive Language-Image Pre-training)")
    st.write("- Variantes testées :")
    st.write("  - **CLIP model A** : sans tokenisation NLTK, avec troncature standard")
    st.write("  - **CLIP model B** : avec tokenisation NLTK et troncature")
    st.write("- Particularités : combinaison vision + langage, meilleure robustesse aux descriptions textuelles")

# ===============================
# SECTION 2 : Résultats comparatifs
# ===============================
st.header("Résultats comparatifs")

# Résultats réels (ton tableau)
results = pd.DataFrame({
    "Scenario": ["BASELINE", "TEST_CLIP_model_A", "TEST_CLIP_model_B"],
    "Description": [
        "VGG16 avec data augmentation (Baseline).",
        "SANS tokenisation NLTK et troncature avec CLIP",
        "AVEC tokenisation NLTK et troncature avec CLIP"
    ],
    "Accuracy_Train": [90.960452, 96.666667, 95.595238],
    "Accuracy_Test": [90.717300, 93.333333, 92.857143],
    "Durée_totale_computation": ["11 min 30 sec", "1 min 35 sec", "1 min 28 sec"]
})

# Affichage du tableau
st.dataframe(results)

# Graphique comparatif des accuracy
fig, ax = plt.subplots(figsize=(5, 3))
results.plot(
    x="Scenario",
    y=["Accuracy_Train", "Accuracy_Test"],
    kind="bar",
    ax=ax,
    legend=False  # on désactive la légende par défaut
)
ax.set_title("Comparaison des performances (Accuracy)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("")  # supprime le label de l'axe X

# Ajouter la légende **à droite** du graphique
ax.legend(["Accuracy_Train", "Accuracy_Test"], loc='center left', bbox_to_anchor=(1, 0.5))

# Centrer le graphique dans le dashboard
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.pyplot(fig, use_container_width=False)

# =============================== 
# SECTION 3 : Visualisations avancées
# ===============================
st.header("Visualisations avancées des performances")

perf_folder = "performances"

# Liste des fichiers de métriques (*.txt mais pas *_cm.txt)
model_files = [f for f in os.listdir(perf_folder) if f.endswith(".txt") and not f.endswith("_cm.txt")]

for model_file in model_files:
    model_name = os.path.splitext(model_file)[0]
    st.subheader(f"Modèle : {model_name}")

    # ----------------------
    # Lecture du fichier metrics
    # ----------------------
    file_path = os.path.join(perf_folder, model_file)
    try:
        lines = open(file_path, "r").readlines()
        rows = []
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("accuracy") or line.startswith("macro") or line.startswith("weighted"):
                continue
            # Split depuis la fin pour récupérer correctement les colonnes
            parts = line.rsplit(maxsplit=4)
            if len(parts) == 5:
                class_name, precision, recall, f1_score, support = parts
                try:
                    rows.append([class_name, float(precision), float(recall), float(f1_score), int(support)])
                except ValueError:
                    # Si conversion échoue, on ignore la ligne
                    continue
        df_metrics = pd.DataFrame(rows, columns=["Classe","Précision","Recall","F1-score","Support"])
        st.write("Métriques par classe")
        st.dataframe(df_metrics)
    except Exception as e:
        st.warning(f"Impossible de lire {model_file}: {e}")

    # ----------------------
    # Matrice de confusion
    # ----------------------
    cm_image_file = os.path.join(perf_folder, f"{model_name}.png")  # ex: baseline.png
    if os.path.exists(cm_image_file):
        st.write("Matrice de confusion")
        st.image(cm_image_file, use_container_width=False, width=1000)  # width ajustable
    else:
        st.info(f"Aucune image de matrice de confusion trouvée pour {model_name}")

# ===============================
# SECTION 4 : Importance des features par catégorie
# ===============================
st.header("Importance des features")

import os

# Dossier où sont stockées tes images
image_folder = "images"

# Récupérer la liste des fichiers d'image
images = os.listdir(image_folder)

# Nettoyer pour enlever l'extension et n'afficher que le nom de catégorie
categories = [os.path.splitext(img)[0] for img in images]

if categories:
    # Sélecteur de catégorie
    selected_category = st.selectbox("Choisissez une catégorie :", categories)

    # Retrouver l'image correspondante
    image_file = selected_category + ".png"  # adapte si tes images sont en .jpg
    image_path = os.path.join(image_folder, image_file)

    # Afficher l'image
    st.subheader(f"Exemple pour la catégorie : {selected_category}")
    st.image(image_path, caption=f"Importance des features - {selected_category}", use_container_width=True)
else:
    st.warning("Aucune image trouvée dans le dossier 'images/'.")

# ===============================
# SECTION 5 : Conclusion
# ===============================
st.header("Conclusion")
st.success("""
-	CLIP surpasse la baseline d’environ 3%.
-	La prise en compte simultanée de l’image et du texte permet de réduire les ambiguïtés.
-	Le zero-shot learning ouvre la voie à une extensibilité sans réentraînement complet.
-	La durée totale de computation est 10 fois plus faible pour CLIP.
""")
