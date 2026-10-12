"""Jauge ludique de variations faciales en temps reel.

Ce programme ne detecte pas les mensonges : il visualise uniquement des
variations de proportions faciales, qui peuvent avoir de nombreuses causes.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
CALIBRATION_SECONDS = 5.0
MIN_CALIBRATION_SAMPLES = 15

# Indices de Face Mesh pour les mesures geometriques.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_EYE_VISUAL = (33, 133, 159, 145)
RIGHT_EYE_VISUAL = (362, 263, 386, 374)
LEFT_EYEBROW_VISUAL = (70, 63, 105, 66)
RIGHT_EYEBROW_VISUAL = (336, 296, 334, 285)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
FACE_LEFT = 234
FACE_RIGHT = 454
EYE_LEFT_CORNER = 33
EYE_RIGHT_CORNER = 263

METRIC_WEIGHTS = {
    "ouverture_yeux": 0.45,
    "ouverture_bouche": 0.40,
    "ecartement_yeux": 0.15,
}
METRIC_TOLERANCES = {
    "ouverture_yeux": 0.04,
    "ouverture_bouche": 0.10,
    "ecartement_yeux": 0.025,
}


def landmark_xy(landmarks, index, frame_width, frame_height):
    """Convertit un point Face Mesh en coordonnees pixel."""
    point = landmarks[index]
    return np.array([point.x * frame_width, point.y * frame_height], dtype=float)


def distance(first, second):
    return float(np.linalg.norm(first - second))


def eye_aspect_ratio(landmarks, indices, frame_width, frame_height):
    """Mesure l'ouverture de l'oeil en la normalisant par sa largeur."""
    points = [landmark_xy(landmarks, i, frame_width, frame_height) for i in indices]
    vertical_one = distance(points[1], points[5])
    vertical_two = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])
    return (vertical_one + vertical_two) / (2.0 * horizontal + 1e-6)


def measure_face(landmarks, frame_width, frame_height):
    """Retourne trois ratios faciaux peu sensibles a la taille du visage."""
    left_eye = eye_aspect_ratio(landmarks, LEFT_EYE, frame_width, frame_height)
    right_eye = eye_aspect_ratio(landmarks, RIGHT_EYE, frame_width, frame_height)

    mouth_width = distance(
        landmark_xy(landmarks, MOUTH_LEFT, frame_width, frame_height),
        landmark_xy(landmarks, MOUTH_RIGHT, frame_width, frame_height),
    )
    mouth_height = distance(
        landmark_xy(landmarks, MOUTH_TOP, frame_width, frame_height),
        landmark_xy(landmarks, MOUTH_BOTTOM, frame_width, frame_height),
    )
    face_width = distance(
        landmark_xy(landmarks, FACE_LEFT, frame_width, frame_height),
        landmark_xy(landmarks, FACE_RIGHT, frame_width, frame_height),
    )
    eye_spacing = distance(
        landmark_xy(landmarks, EYE_LEFT_CORNER, frame_width, frame_height),
        landmark_xy(landmarks, EYE_RIGHT_CORNER, frame_width, frame_height),
    )

    return {
        "ouverture_yeux": (left_eye + right_eye) / 2.0,
        "ouverture_bouche": mouth_height / (mouth_width + 1e-6),
        "ecartement_yeux": eye_spacing / (face_width + 1e-6),
    }


def draw_face_landmarks(frame, landmarks, frame_width, frame_height):
    """Dessine les yeux, les sourcils et les points de reference faciaux."""
    features = [
        (LEFT_EYEBROW_VISUAL, (245, 245, 245), False),
        (RIGHT_EYEBROW_VISUAL, (245, 245, 245), False),
        (LEFT_EYE_VISUAL, (0, 255, 255), False),
        (RIGHT_EYE_VISUAL, (0, 255, 255), False),
    ]

    for indices, color, closed in features:
        points = [landmark_xy(landmarks, index, frame_width, frame_height).astype(int) for index in indices]
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, 2)
        for point in points:
            cv2.circle(frame, tuple(point), 2, color, -1)

    for index in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
        point = landmark_xy(landmarks, index, frame_width, frame_height).astype(int)
        cv2.circle(frame, tuple(point), 2, (70, 220, 180), -1)


def draw_interface(frame, score, metrics, baseline, calibrating, seconds_left, face_found):
    """Dessine l'etat, les mesures et une jauge de couleur sur l'image."""
    height, width = frame.shape[:2]
    panel_height = min(190, height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (width - 12, panel_height), (20, 24, 29), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if calibrating:
        status = f"Calibration neutre : {seconds_left:.1f} s"
        score_text = "--"
    elif not face_found:
        status = "Aucun visage detecte"
        score_text = f"{int(score):02d}"
    else:
        status = "Indice de variation (pas un detecteur de mensonge)"
        score_text = f"{int(score):02d} / 100"

    cv2.putText(frame, "JAUGE LUDIQUE", (28, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (245, 245, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, status, (28, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (210, 215, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, score_text, (width - 190, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (245, 245, 240), 2, cv2.LINE_AA)

    gauge_left, gauge_right = 28, max(29, width - 28)
    gauge_top, gauge_bottom = 91, 108
    cv2.rectangle(frame, (gauge_left, gauge_top), (gauge_right, gauge_bottom),
                  (75, 78, 82), -1)
    if not calibrating:
        fill_right = gauge_left + int((gauge_right - gauge_left) * np.clip(score, 0, 100) / 100)
        if score < 35:
            color = (80, 190, 100)
        elif score < 70:
            color = (30, 185, 235)
        else:
            color = (65, 75, 235)
        cv2.rectangle(frame, (gauge_left, gauge_top), (fill_right, gauge_bottom), color, -1)

    metric_y = 137
    if baseline is None:
        metric_line = "Regardez la camera et gardez une expression naturelle."
    elif metrics is not None:
        metric_line = (
            f"Yeux {metrics['ouverture_yeux']:.2f}   "
            f"Bouche {metrics['ouverture_bouche']:.2f}   "
            f"Ecart yeux {metrics['ecartement_yeux']:.2f}"
        )
    else:
        metric_line = "Mesures en attente du visage..."
    cv2.putText(frame, metric_line, (28, metric_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (225, 228, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q ou Echap : quitter", (28, 163), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (175, 180, 185), 1, cv2.LINE_AA)


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            "Webcam inaccessible. Verifiez CAMERA_INDEX et les autorisations camera."
        )

    calibration_started = time.monotonic()
    calibration_samples = []
    baseline = None
    score = 0.0
    score_history = deque(maxlen=5)

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
                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_frame)
                metrics = None
                face_found = bool(result.multi_face_landmarks)
                elapsed = time.monotonic() - calibration_started
                calibrating = baseline is None

                if face_found:
                    face_landmarks = result.multi_face_landmarks[0].landmark
                    metrics = measure_face(face_landmarks, frame_width, frame_height)

                    draw_face_landmarks(frame, face_landmarks, frame_width, frame_height)

                    if calibrating:
                        calibration_samples.append(metrics)
                        if elapsed >= CALIBRATION_SECONDS and len(calibration_samples) >= MIN_CALIBRATION_SAMPLES:
                            baseline = {
                                name: float(np.median([sample[name] for sample in calibration_samples]))
                                for name in METRIC_WEIGHTS
                            }
                    else:
                        deviations = {}
                        for name in METRIC_WEIGHTS:
                            raw_delta = abs(metrics[name] - baseline[name])
                            deviations[name] = min(1.0, raw_delta / METRIC_TOLERANCES[name])
                        current_score = 100.0 * sum(
                            METRIC_WEIGHTS[name] * deviations[name]
                            for name in METRIC_WEIGHTS
                        )
                        score_history.append(current_score)
                        score = float(np.mean(score_history))

                seconds_left = max(0.0, CALIBRATION_SECONDS - elapsed)
                draw_interface(frame, score, metrics, baseline,
                               calibrating, seconds_left, face_found)
                cv2.imshow("Jauge faciale ludique", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
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
