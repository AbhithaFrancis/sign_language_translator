
import cv2
import mediapipe as mp
import numpy as np
import joblib
import subprocess
import threading


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("sign_model.pkl")


# ============================================================
# SPEECH
# ============================================================

def speak(text):

    print("VOICE:", text)

    powershell_command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speak.Rate = 0; "
        "$speak.Volume = 100; "
        f'$speak.Speak("{text}");'
        "$speak.Dispose();"
    )

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            powershell_command
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )


def speak_in_background(text):

    thread = threading.Thread(
        target=speak,
        args=(text,),
        daemon=True
    )

    thread.start()


# ============================================================
# SPEECH MAP
# ============================================================

speech_map = {
    "HELLO": "Hello",
    "THANK_YOU": "Thank you",
    "YES": "Yes",
    "NO": "No",
    "STOP": "Stop",
    "LOVE": "Love",
    "FRIEND": "Friend",
    "BYE": "Bye"
}


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(row):

    landmarks = np.array(
        row,
        dtype=float
    ).reshape(21, 3)

    # Wrist = origin
    wrist = landmarks[0]

    landmarks = landmarks - wrist

    # Scale normalization
    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

last_spoken_sign = None

CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# START LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp += 33


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        display_text = "Show a sign"


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # EXTRACT 21 LANDMARKS
            # ------------------------------------------------

            landmarks = []

            for point in hand:

                landmarks.extend([
                    point.x,
                    point.y,
                    point.z
                ])


            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            features = normalize_landmarks(
                landmarks
            )

            features = features.reshape(
                1,
                -1
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                features
            )[0]


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = np.max(
                probabilities
            )


            # =================================================
            # CONFIDENCE ≥ 90%
            # =================================================

            if confidence >= CONFIDENCE_THRESHOLD:

                display_text = (
                    f"{prediction} "
                    f"{confidence * 100:.1f}%"
                )


                # ------------------------------------------------
                # NEW SIGN
                # ------------------------------------------------

                if prediction != last_spoken_sign:

                    spoken_word = speech_map.get(
                        prediction,
                        prediction.replace(
                            "_",
                            " "
                        ).title()
                    )


                    print()
                    print(
                        "Prediction:",
                        prediction
                    )

                    print(
                        "Confidence:",
                        f"{confidence * 100:.2f}%"
                    )

                    print(
                        "Speaking:",
                        spoken_word
                    )


                    # SPEAK
                    speak_in_background(
                        spoken_word
                    )


                    # Remember this sign
                    last_spoken_sign = prediction


            else:

                display_text = (
                    f"Not sure "
                    f"{confidence * 100:.1f}%"
                )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            display_text = "Show a sign"

            # Reset
            last_spoken_sign = None


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            display_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "Sign Language Recognition",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Program ended.")

