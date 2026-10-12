"""Affiche en direct la sortie du modele entraine sur les etiquettes collectees."""

import argparse
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "modele_mensonge.pkl"
FEATURES = ["left_eye_ear", "mouth_opening", "blink_motion_ratio"]
LEFT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE_VISUAL = (33, 133, 159, 145)
RIGHT_EYE_VISUAL = (362, 263, 386, 374)
LEFT_EYEBROW_VISUAL = (70, 63, 105, 66)
RIGHT_EYEBROW_VISUAL = (336, 296, 334, 285)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def point(landmarks, index, width, height):
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def euclidean_distance(first, second):
    return float(np.linalg.norm(first - second))


def eye_aspect_ratio(landmarks, indices, width, height):
    points = [point(landmarks, index, width, height) for index in indices]
    vertical_one = euclidean_distance(points[1], points[5])
    vertical_two = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])
    return (vertical_one + vertical_two) / (2.0 * horizontal + 1e-6)


def extract_metrics(landmarks, width, height, previous_ear):
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
    mouth_width = euclidean_distance(
        point(landmarks, MOUTH_LEFT, width, height),
        point(landmarks, MOUTH_RIGHT, width, height),
    )
    mouth_height = euclidean_distance(
        point(landmarks, MOUTH_TOP, width, height),
        point(landmarks, MOUTH_BOTTOM, width, height),
    )
    blink_motion_ratio = 0.0 if previous_ear is None else (
        abs(left_ear - previous_ear) / max(previous_ear, 1e-6)
    )
    return {
        "left_eye_ear": left_ear,
        "mouth_opening": mouth_height / (mouth_width + 1e-6),
        "blink_motion_ratio": blink_motion_ratio,
    }


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}. Entrainez-le d'abord avec train.py."
        )
    saved = joblib.load(model_path)
    if isinstance(saved, dict):
        classifier = saved.get("model")
        saved_features = saved.get("features")
        if classifier is None or saved_features != FEATURES:
            raise ValueError("Le fichier modele ne contient pas les caracteristiques attendues.")
    else:
        classifier = saved

    if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
        raise ValueError("Le modele doit fournir predict_proba et les classes 0/1.")
    if set(classifier.classes_) != {0, 1}:
        raise ValueError("Le modele doit avoir ete entraine avec les classes 0 et 1.")
    return classifier


def draw_overlay(frame, metrics, probability, face_found):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (width - 12, 174), (20, 24, 29), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.putText(frame, "PREDICTION DU MODELE", (26, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 240), 2, cv2.LINE_AA)

    if not face_found:
        status = "Aucun visage detecte"
        color = (210, 215, 220)
        probability_text = "Probabilite: --"
    else:
        predicted_lie = probability >= 0.5
        status = "MENSONGE" if predicted_lie else "VERITE"
        color = (45, 55, 235) if predicted_lie else (70, 205, 100)
        probability_text = f"Probabilite MENSONGE: {probability:.1%}"

    cv2.putText(frame, status, (26, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, probability_text, (26, 111), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)

    left, right = 26, width - 26
    top, bottom = 124, 142
    cv2.rectangle(frame, (left, top), (right, bottom), (75, 78, 82), -1)
    if face_found:
        fill_right = left + int((right - left) * np.clip(probability, 0.0, 1.0))
        cv2.rectangle(frame, (left, top), (fill_right, bottom), color, -1)

    if metrics is not None:
        metric_text = (
            f"EAR gauche {metrics['left_eye_ear']:.3f}   "
            f"Bouche {metrics['mouth_opening']:.3f}   "
            f"Mouvement {metrics['blink_motion_ratio']:.3f}"
        )
    else:
        metric_text = "En attente de mesures..."
    cv2.putText(frame, metric_text, (26, 162), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (210, 215, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Sortie experimentale basee sur vos etiquettes; pas un detecteur fiable.",
                (26, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (225, 225, 225), 1, cv2.LINE_AA)


def draw_face_landmarks(frame, landmarks, width, height):
    features = [
        (LEFT_EYEBROW_VISUAL, (245, 245, 245), False),
        (RIGHT_EYEBROW_VISUAL, (245, 245, 245), False),
        (LEFT_EYE_VISUAL, (0, 255, 255), False),
        (RIGHT_EYE_VISUAL, (0, 255, 255), False),
    ]

    for indices, color, closed in features:
        points = [point(landmarks, index, width, height).astype(int) for index in indices]
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, 2)
        for point in points:
            cv2.circle(frame, tuple(point), 2, color, -1)

    for index in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
        xy = point(landmarks, index, width, height).astype(int)
        cv2.circle(frame, tuple(xy), 3, (70, 220, 180), -1)


def run(model_path, camera_index=0):
    classifier = load_model(model_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Webcam inaccessible. Verifiez l'index et les autorisations camera.")

    previous_ear = None
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Impossible de lire une image depuis la webcam.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_found = bool(result.multi_face_landmarks)
                metrics = None
                probability = 0.0

                if face_found:
                    landmarks = result.multi_face_landmarks[0].landmark
                    metrics = extract_metrics(landmarks, width, height, previous_ear)
                    previous_ear = metrics["left_eye_ear"]
                    feature_row = np.array([[metrics[name] for name in FEATURES]], dtype=float)
                    probabilities = classifier.predict_proba(feature_row)[0]
                    positive_index = list(classifier.classes_).index(1)
                    probability = float(probabilities[positive_index])

                    draw_face_landmarks(frame, landmarks, width, height)
                else:
                    previous_ear = None

                draw_overlay(frame, metrics, probability, face_found)
                cv2.imshow("Classification faciale experimentale", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Modele joblib a charger (defaut : {DEFAULT_MODEL})")
    parser.add_argument("--camera", type=int, default=0, help="Index de la webcam (defaut : 0)")
    args = parser.parse_args()
    run(args.model, args.camera)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Erreur : {error}") from error
