"""Collecte de mesures faciales etiquetees dans dataset.csv.

Les etiquettes sont saisies par l'utilisateur et ne constituent pas une
verification de la veracite d'une declaration.
"""

import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DATASET_PATH = Path(__file__).resolve().with_name("dataset.csv")
FEATURES = ("left_eye_ear", "mouth_opening", "blink_motion_ratio")
LABELS = {ord("v"): 0, ord("m"): 1}
LABEL_NAMES = {0: "VERITE", 1: "MENSONGE"}

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


def load_counts(csv_path):
    counts = {0: 0, 1: 0}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError):
                continue
            if label in counts:
                counts[label] += 1
    return counts


def ensure_csv_header(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=(*FEATURES, "label"))
            writer.writeheader()


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


def draw_hud(frame, metrics, counts, face_found):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (width - 12, 160), (20, 24, 29), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.putText(frame, "COLLECTE DES MESURES FACIALES", (26, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 240), 2, cv2.LINE_AA)

    if face_found and metrics is not None:
        metric_text = (
            f"EAR gauche: {metrics['left_eye_ear']:.3f}   "
            f"Bouche: {metrics['mouth_opening']:.3f}   "
            f"Mouvement: {metrics['blink_motion_ratio']:.3f}"
        )
    else:
        metric_text = "Aucun visage detecte - placez votre visage devant la camera."
    cv2.putText(frame, metric_text, (26, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (220, 225, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"VERITE (v): {counts[0]}    MENSONGE (m): {counts[1]}",
                (26, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 220, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, "v/m: enregistrer l'image courante | q/Echap: quitter",
                (26, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 195, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Etiquettes manuelles, pas une detection fiable du mensonge.",
                (26, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 180), 1, cv2.LINE_AA)


def main():
    ensure_csv_header(DATASET_PATH)
    counts = load_counts(DATASET_PATH)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam inaccessible. Verifiez les autorisations camera.")

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

                if face_found:
                    landmarks = result.multi_face_landmarks[0].landmark
                    metrics = extract_metrics(landmarks, width, height, previous_ear)
                    previous_ear = metrics["left_eye_ear"]
                    draw_face_landmarks(frame, landmarks, width, height)
                else:
                    previous_ear = None

                draw_hud(frame, metrics, counts, face_found)
                cv2.imshow("Collecte dataset", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in LABELS:
                    if not face_found or metrics is None:
                        print("Aucun visage: aucune ligne enregistree.")
                        continue
                    label = LABELS[key]
                    with DATASET_PATH.open("a", newline="", encoding="utf-8") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=(*FEATURES, "label"))
                        writer.writerow({**metrics, "label": label})
                    counts[label] += 1
                    print(f"{LABEL_NAMES[label]} enregistre dans {DATASET_PATH.name}.")
                elif key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Erreur : {error}")
    except KeyboardInterrupt:
        print("Arret demande.")
