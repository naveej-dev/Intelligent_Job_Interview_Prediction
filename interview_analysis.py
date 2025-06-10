import cv2
import math
import numpy as np
import pandas as pd
from fer import FER
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime

def run_interview_analysis():
    model = load_model('./model_vgg16_finetuned.keras')
    picture_size = 48

    detector = FER(mtcnn=True)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)

    cap = cv2.VideoCapture(0)
    final_scores = []
    frame_logs = []

    def assess_upper_body_posture(landmarks):
        ls = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
        nose = landmarks[mp_holistic.PoseLandmark.NOSE]

        shoulder_diff = abs(ls.y - rs.y)
        shoulders_uneven = shoulder_diff > 0.05

        spine_top = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2]
        spine_bottom = [(lh.x + rh.x) / 2, (lh.y + rh.y) / 2]
        dx = spine_top[0] - spine_bottom[0]
        dy = spine_top[1] - spine_bottom[1]
        spine_angle = abs(math.degrees(math.atan2(dy, dx)))
        slouching = spine_angle < 75

        shoulder_center_x = (ls.x + rs.x) / 2
        head_shift = abs(nose.x - shoulder_center_x)
        head_leaning = head_shift > 0.08

        chin_to_chest = nose.y > ls.y + 0.1 or nose.y > rs.y + 0.1
        shoulder_to_face_distance = abs(ls.y - nose.y) < 0.1 or abs(rs.y - nose.y) < 0.1

        if chin_to_chest or shoulder_to_face_distance or slouching or shoulders_uneven or head_leaning:
            posture = "Informal"
            color = (0, 0, 255)
            posture_score = -5
        else:
            posture = "Formal"
            color = (0, 255, 0)
            posture_score = 5

        return posture, color, posture_score

    def detect_gesture(landmarks, label):
        wrist = landmarks[mp_holistic.HandLandmark.WRIST]
        index_tip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]

        gesture_score = 10
        if index_tip.y < 0.5:
            gesture = "Informal"
            gesture_score = -5
            color = (0, 0, 255)
        else:
            gesture = "Formal"
            gesture_score = 5
            color = (0, 255, 0)

        cv2.putText(frame, f"{label} Gesture: {gesture}",
                    (30, 120 if label == "Right" else 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return gesture, gesture_score

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emotion_results = detector.detect_emotions(frame)
        emotion_score = 5
        dominant_emotion = "unknown"
        if emotion_results:
            emotions = emotion_results[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            if dominant_emotion in ["sad", "angry"]:
                emotion_color = (0, 0, 255)
                emotion_score = -5
            elif dominant_emotion == "happy":
                emotion_color = (0, 255, 0)
                emotion_score = 8
            elif dominant_emotion in ["calm", "neutral"]:
                emotion_color = (0, 255, 0)
                emotion_score = 5
            else:
                emotion_color = (0, 255, 0)
                emotion_score = 1

            cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        posture_label = "Unknown"
        posture_score = 10
        posture_color = (0, 255, 0)
        if results.pose_landmarks:
            posture_label, posture_color, posture_score = assess_upper_body_posture(results.pose_landmarks.landmark)
            cv2.putText(frame, f"Posture: {posture_label}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2)
            )

        gesture_right = "None"
        gesture_score_right = 10
        if results.right_hand_landmarks:
            gesture_right, gesture_score_right = detect_gesture(results.right_hand_landmarks.landmark, "Right")
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        gesture_left = "None"
        gesture_score_left = 10
        if results.left_hand_landmarks:
            gesture_left, gesture_score_left = detect_gesture(results.left_hand_landmarks.landmark, "Left")
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        img = cv2.resize(frame, (picture_size, picture_size))
        img_array = np.expand_dims(img, axis=0) / 255.0
        cnn_pred = model.predict(img_array, verbose=0)[0][0]
        cnn_conf = cnn_pred * 100

        if cnn_conf >= 60:
            cnn_label = "Confident"
            cnn_color = (0, 255, 0)
        elif 40 <= cnn_conf < 60:
            cnn_label = "Neutral"
            cnn_color = (0, 255, 255)
        else:
            cnn_label = "Non-Confident"
            cnn_color = (0, 0, 255)

        cv2.putText(frame, f"Confidence: {cnn_label} ({cnn_conf:.1f}%)", (30, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cnn_color, 2)

        raw_score = emotion_score + posture_score + (gesture_score_right + gesture_score_left) / 2
        normalized_score = (raw_score / 28) * 100
        final_confidence_score = (cnn_conf + normalized_score) / 2

        final_scores.append(final_confidence_score)

        cv2.putText(frame, f"Overall Score: {final_confidence_score:.1f}/100", (30, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_logs.append({
            "Dominant Emotion": dominant_emotion,
            "Emotion Score": emotion_score,
            "Posture": posture_label,
            "Posture Score": posture_score,
            "Gesture Left": gesture_left,
            "Gesture Left Score": gesture_score_left,
            "Gesture Right": gesture_right,
            "Gesture Right Score": gesture_score_right,
            "CNN Confidence (%)": cnn_conf,
            "CNN Label": cnn_label,
            "Interview Score": final_confidence_score
        })

        cv2.imshow('Interview Analysis (Emotion + Posture + Gesture + CNN)', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if final_scores:
        mean_score = sum(final_scores) / len(final_scores)
    else:
        mean_score = 0

    if mean_score >= 60:
        mean_cnn_label = "Confident"
    elif mean_score >= 40:
        mean_cnn_label = "Neutral"
    else:
        mean_cnn_label = "Non-Confident"

    print(f"\nFinal Interview Score: {mean_score:.1f}%")
    if mean_score < 70:
        print("You need to improve your performance.")
    else:
        print("You are doing great! Keep it up.")

    # Save to CSV with timestamp
    df = pd.DataFrame(frame_logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_analysis_log_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Frame-wise interview data saved to '{filename}'.")

    return mean_score, mean_cnn_label
