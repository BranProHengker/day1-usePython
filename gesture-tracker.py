import cv2
import mediapipe as mp
import numpy as np
import os
from math import hypot

IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "THINKING": "thinking.jpg",
    "PRAY": "pray.jpg",
    "KAGET": "kaget.jpg",
    "MELET": "melet-wleee.jpg",
    "IMUT": "imut.jpg",
    "NEUTRAL": "neutral.jpg",
}

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Hands: izinkan 2 tangan buat gesture PRAY
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Face mesh untuk ekspresi
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,  # True kalau mau landmark bibir/iris lebih detail (lebih berat)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_and_resize_image(path, target_height):
    full_path = os.path.join(os.getcwd(), path)
    img = cv2.imread(full_path)
    if img is None:
        print(f"[WARN] Gagal load image: {path}")
        return None
    ratio = target_height / img.shape[0]
    target_width = int(img.shape[1] * ratio)
    return cv2.resize(img, (target_width, target_height))

# ---------- UTIL FACE ----------
# beberapa id penting (mediapipe facemesh)
MOUTH_UP = 13
MOUTH_DOWN = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# kiri = dari user (sebenarnya kanan kamera). Ini cukup stabil untuk eye open ratio
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

def l2(a, b, W, H):
    ax, ay = int(a.x * W), int(a.y * H)
    bx, by = int(b.x * W), int(b.y * H)
    return hypot(ax - bx, ay - by)

def mouth_open_ratio(face_lms, W, H):
    up = face_lms.landmark[MOUTH_UP]
    down = face_lms.landmark[MOUTH_DOWN]
    left = face_lms.landmark[MOUTH_LEFT]
    right = face_lms.landmark[MOUTH_RIGHT]
    height = l2(up, down, W, H)
    width = l2(left, right, W, H) + 1e-6
    return height / width  # semakin besar = semakin mangap

def eye_open_ratio(face_lms, top, bottom, left, right, W, H):
    v = l2(face_lms.landmark[top], face_lms.landmark[bottom], W, H)
    h = l2(face_lms.landmark[left], face_lms.landmark[right], W, H) + 1e-6
    return v / h  # kecil = merem/pejam

# ---------- UTIL HAND ----------
def classify_hand_gesture(hand_lms):
    y_thumb_tip = hand_lms.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    y_index_tip = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    y_middle_tip = hand_lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    y_ring_tip = hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    y_pinky_tip = hand_lms.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    y_middle_pip = hand_lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    is_thumb_up = y_thumb_tip < y_middle_pip
    are_others_down = (
        y_index_tip > y_middle_pip and
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )
    if is_thumb_up and are_others_down:
        return "THUMBS_UP"

    is_index_up = y_index_tip < y_middle_pip
    is_other_fingers_down = (
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )
    is_thumb_down = y_thumb_tip > y_middle_pip
    if is_index_up and is_other_fingers_down and is_thumb_down:
        return "POINTING"

    return "NEUTRAL"

def is_thinking(hand_lms, face_lms, W, H, max_px_dist=50):
    if not (hand_lms and face_lms):
        return False
    index_tip = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    nose_tip = face_lms.landmark[4]  # hidung
    dist = l2(index_tip, nose_tip, W, H)

    y_middle_pip = hand_lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    y_middle_tip = hand_lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_down = y_middle_tip > y_middle_pip

    return dist < max_px_dist and middle_down

def is_pray(two_hand_lms, W, H, max_px_dist=80):
    # butuh 2 tangan: cek jarak antara ujung telunjuk kiri & kanan
    if len(two_hand_lms) < 2:
        return False
    h1, h2 = two_hand_lms[0], two_hand_lms[1]
    i1 = h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    i2 = h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    d = l2(i1, i2, W, H)
    return d < max_px_dist

def classify_face_expressions(face_lms, W, H):
    """Kembalikan salah satu: KAGET, MELET, IMUT, atau None"""
    if not face_lms:
        return None

    mratio = mouth_open_ratio(face_lms, W, H)  # ~0.1 tertutup, >0.6 sangat terbuka
    le = eye_open_ratio(face_lms, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT, W, H)
    re = eye_open_ratio(face_lms, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, W, H)

    # ambang bisa kamu tweak sesuai kamera
    if mratio > 0.60:
        return "KAGET"
    if 0.35 <= mratio <= 0.60:
        return "MELET"
    # wink/imut: satu mata "kecil", satu "besar"
    if (le < 0.18 and re > 0.22) or (re < 0.18 and le > 0.22):
        return "IMUT"

    return None

# --- MAIN PROGRAM ---
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Yang Gerak Monyet. Tekan 'q' untuk keluar.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # KUMPULKAN LANDMARKS
    multi_hands = hand_results.multi_hand_landmarks or []
    face_lms = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

    gesture = "NEUTRAL"

    # 1) PRAY (dua tangan rapat)
    if len(multi_hands) >= 2 and is_pray(multi_hands, W, H):
        gesture = "PRAY"

    # 2) THINKING (tangan + wajah)
    if gesture == "NEUTRAL" and len(multi_hands) >= 1 and face_lms:
        if is_thinking(multi_hands[0], face_lms, W, H):
            gesture = "THINKING"

    # 3) Hand-only (thumbs up / pointing)
    if gesture == "NEUTRAL" and len(multi_hands) >= 1:
        gesture = classify_hand_gesture(multi_hands[0])

    # 4) Face-only (kaget / melet / imut)
    if gesture == "NEUTRAL":
        face_exp = classify_face_expressions(face_lms, W, H)
        if face_exp:
            gesture = face_exp

    # DRAW debugging (opsional)
    for hlms in multi_hands:
        mp_drawing.draw_landmarks(
            frame, hlms, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        )
    if face_lms:
        mp_drawing.draw_landmarks(
            frame, face_lms, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(thickness=1)
        )

    # Tampilkan gambar sample gesture di sisi kanan
    right_img = load_and_resize_image(IMAGE_PATHS.get(gesture, "neutral.jpg"), H)
    if right_img is not None:
        out = np.concatenate((frame, right_img), axis=1)
        cv2.putText(out, f"Gesture: {gesture}", (W + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        out = frame
        cv2.putText(out, f"Gesture: {gesture} (image not found)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Gesture & Image Pairing", out)
    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        break

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
