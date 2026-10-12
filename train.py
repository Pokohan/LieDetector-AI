"""Entraine un classifieur a partir de dataset.csv et le sauvegarde avec joblib."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset.csv"
DEFAULT_MODEL = BASE_DIR / "modele_mensonge.pkl"
FEATURES = ["left_eye_ear", "mouth_opening", "blink_motion_ratio"]
LABEL_MAP = {"0": 0, "1": 1, "VERITE": 0, "MENSONGE": 1}


def load_dataset(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {dataset_path}. Lancez d'abord main.py et collectez des donnees."
        )

    data = pd.read_csv(dataset_path)
    required_columns = [*FEATURES, "label"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le CSV : {', '.join(missing_columns)}")

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data["label"] = data["label"].astype(str).str.upper().map(LABEL_MAP)
    data = data.dropna(subset=required_columns)
    finite_rows = np.isfinite(data[FEATURES].to_numpy(dtype=float)).all(axis=1)
    data = data.loc[finite_rows].copy()
    data["label"] = data["label"].astype(int)

    if data.empty:
        raise ValueError("Le CSV ne contient aucune ligne valide.")
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Le dataset doit contenir les deux etiquettes 0 (VERITE) et 1 (MENSONGE).")
    if data["label"].value_counts().min() < 2:
        raise ValueError("Collectez au moins deux exemples pour chaque categorie.")
    return data


def train(dataset_path, model_path):
    data = load_dataset(dataset_path)
    features = data[FEATURES]
    labels = data["label"]

    # Reserve au moins un exemple de chaque classe pour l'evaluation.
    test_count = max(labels.nunique(), int(np.ceil(0.25 * len(data))))
    test_count = min(test_count, len(data) - labels.nunique())
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_count,
        random_state=42,
        stratify=labels,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    print(f"Accuracy sur le jeu de test : {accuracy:.2%}")
    print(f"Exemples utilises : {len(data)} (test : {len(y_test)})")

    # Le modele enregistre est ajuste sur toutes les donnees disponibles.
    classifier.fit(features, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier, "features": FEATURES}, model_path)
    print(f"Modele sauvegarde : {model_path}")
    print("Cette accuracy mesure l'accord avec les etiquettes collectees, pas la detection du mensonge.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"CSV source (defaut : {DEFAULT_DATASET})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Fichier de sortie .pkl (defaut : {DEFAULT_MODEL})")
    args = parser.parse_args()
    train(args.dataset, args.model)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Erreur : {error}") from error
