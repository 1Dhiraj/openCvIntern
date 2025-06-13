import cv2
import mediapipe as mp
import webbrowser
import time
import pyttsx3
import numpy as np
import subprocess

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Cooldown settings
last_open_time = 0
last_speech_time = 0  # Track last time voice was played
cooldown = 3  # seconds

# Load the flaming skull PNG (ensure it has an alpha channel for transparency)
flaming_skull = cv2.imread("flaming_skull.png", cv2.IMREAD_UNCHANGED)
flame_red = cv2.imread("flaming_red.png", cv2.IMREAD_UNCHANGED)
if flaming_skull is None:
    print("❌ Failed to load flaming_skull.png")
    exit()
if flame_red is None:
    print("❌ Failed to load flaming_red.png")
    exit()

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def speak(text):
    """Speak the given text with cooldown to avoid repetition."""
    global last_speech_time
    if time.time() - last_speech_time > cooldown:
        engine.say(text)
        engine.runAndWait()
        last_speech_time = time.time()

def overlay_image_alpha(img, img_overlay, x, y, width, height):
    """Overlay an image with an alpha channel onto the frame."""
    # Resize the overlay image to fit the face
    img_overlay = cv2.resize(img_overlay, (width, height))

    # Split the overlay into color and alpha channels
    if img_overlay.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_mask = img_overlay[:, :, 3] / 255.0
        img_overlay = img_overlay[:, :, :3]
    else:
        alpha_mask = np.ones((img_overlay.shape[0], img_overlay.shape[1]))
    
    # Calculate the region of interest (ROI) in the frame
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    
    # Adjust the overlay and alpha mask to fit the ROI
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Extract the ROI from the frame
    roi = img[y1:y2, x1:x2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    # Blend the overlay with the ROI
    for c in range(0, 3):
        roi[:, :, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                        alpha_inv * roi[:, :, c])

    img[y1:y2, x1:x2] = roi
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    hand_result = hands.process(rgb_frame)
    finger_count = 0

    if hand_result.multi_hand_landmarks:
        print(f"✅ Detected {len(hand_result.multi_hand_landmarks)} hand(s)")
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_count = count_fingers(hand_landmarks)
            print("👉 Fingers up:", finger_count)

            # Show finger count on screen
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Voice feedback and open URL for 1 finger
            if finger_count == 1 and (time.time() - last_open_time) > cooldown:
                speak("Rick Roll alert")
                print(" Opening YouTube...")
                webbrowser.open("https://www.youtube.com/watch?v=xvFZjo5PgG0&list=RDxvFZjo5PgG0&start_radio=1")
                last_open_time = time.time()

    else:
        print("🛑 No hands detected")

    # Process face detection
    face_result = face_detection.process(rgb_frame)

    # Overlay the flaming skull on the face if 2 fingers are detected
    if finger_count == 2 and face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)

            # Adjust the position and size to center the flaming skull on the face
            skull_width = int(w * 1.5)  # Scale the skull to be slightly larger than the face
            skull_height = int(h * 1.5)
            x_offset = x - (skull_width - w) // 2
            y_offset = y - (skull_height - h) // 2

            # Overlay the flaming skull on the face
            frame = overlay_image_alpha(frame, flaming_skull, x_offset, y_offset, skull_width, skull_height)

    # Overlay the red flaming skull on the face if 3 fingers are detected
    if finger_count == 3 and face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)

            # Adjust the position and size to center the red flaming skull on the face
            skull_width = int(w * 2.0)  # Adjusted scale for better proportion with the new image
            skull_height = int(h * 3.0)  # Taller to account for the torso in the new image
            x_offset = x - (skull_width - w) // 2
            y_offset = y - (skull_height - h) // 2 - 30  # Adjusted offset for better centering

            # Overlay the red flaming skull on the face
            frame = overlay_image_alpha(frame, flame_red, x_offset, y_offset, skull_width, skull_height)
    
    if finger_count == 4 and (time.time() - last_open_time) > cooldown:
        speak("Pushing code to GitHub")
        print("📤 Pushing code to GitHub...")
        try:
            # Git commands: add, commit, and push
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Auto-commit from gesture detection"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)  # Replace 'main' with your branch if different
            print("✅ Successfully pushed to GitHub")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to push to GitHub: {e}")
        last_open_time = time.time()

    cv2.imshow("Hand Gesture Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()