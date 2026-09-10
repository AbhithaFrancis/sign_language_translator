import cv2
import mediapipe as mp
import csv
import os
import time

# -----------------------------------
# MediaPipe setup
# -----------------------------------

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

landmarker = HandLandmarker.create_from_options(options)

# -----------------------------------
# Camera
# -----------------------------------

cap = cv2.VideoCapture(0)

# -----------------------------------
# CSV setup
# -----------------------------------

file_exists = os.path.exists("sign_data.csv")

file = open("sign_data.csv", "a", newline="")
writer = csv.writer(file)

if not file_exists:

    header = []

    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")

    header.append("label")

    writer.writerow(header)

# -----------------------------------
# Dataset settings
# -----------------------------------

current_label = None
sample_count = 0

max_samples = 100

# MediaPipe video timestamp
timestamp = 0


# -----------------------------------
# Instructions
# -----------------------------------

print("---------------------------------------")
print(" SIGN LANGUAGE DATA COLLECTION")
print("---------------------------------------")

print("H = HELLO")
print("Y = YES")
print("N = NO")
print("T = THANK YOU")
print("S = STOP")
print("L = LOVE")
print("F = FRIEND")
print("B = BYE")

print("Q = QUIT")

print("---------------------------------------")
print("Show BOTH palm and back of hand")
print("---------------------------------------")


# -----------------------------------
# Main loop
# -----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp
    timestamp += 33

    # Detect hand
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # -----------------------------------
    # Hand detected
    # -----------------------------------

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # Draw all 21 landmarks
        for i, landmark in enumerate(landmarks):

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            # Draw landmark
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # -----------------------------------
        # Draw connections
        # -----------------------------------

        connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (17,18), (18,19), (19,20),
            (0,17)
        ]

        for start, end in connections:

            x1 = int(
                landmarks[start].x *
                frame.shape[1]
            )

            y1 = int(
                landmarks[start].y *
                frame.shape[0]
            )

            x2 = int(
                landmarks[end].x *
                frame.shape[1]
            )

            y2 = int(
                landmarks[end].y *
                frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # -----------------------------------
        # Collect dataset
        # -----------------------------------

        if current_label is not None:

            row = []

            for landmark in landmarks:

                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            row.append(current_label)

            writer.writerow(row)

            sample_count += 1

            print(
                f"{current_label}: "
                f"{sample_count}/{max_samples}"
            )

            # -----------------------------------
            # Automatically stop
            # -----------------------------------

            if sample_count >= max_samples:

                print(
                    f"\n{current_label} COMPLETE! ✅\n"
                )

                current_label = None
                sample_count = 0

    # -----------------------------------
    # Display status
    # -----------------------------------

    if current_label:

        cv2.putText(
            frame,
            f"Collecting: {current_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{max_samples}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Press H/Y/N/T/S/L/F/B",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Palm + Back of hand",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------------
    # Show camera
    # -----------------------------------

    cv2.imshow(
        "Sign Language Dataset",
        frame
    )

    # -----------------------------------
    # Keyboard
    # -----------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("h"):

        current_label = "HELLO"
        sample_count = 0

        print("\nCollecting HELLO...")

    elif key == ord("y"):

        current_label = "YES"
        sample_count = 0

        print("\nCollecting YES...")

    elif key == ord("n"):

        current_label = "NO"
        sample_count = 0

        print("\nCollecting NO...")

    elif key == ord("t"):

        current_label = "THANK_YOU"
        sample_count = 0

        print("\nCollecting THANK YOU...")

    elif key == ord("s"):

        current_label = "STOP"
        sample_count = 0

        print("\nCollecting STOP...")

    elif key == ord("l"):

        current_label = "LOVE"
        sample_count = 0

        print("\nCollecting LOVE...")

    elif key == ord("f"):

        current_label = "FRIEND"
        sample_count = 0

        print("\nCollecting FRIEND...")

    elif key == ord("b"):

        current_label = "BYE"
        sample_count = 0

        print("\nCollecting BYE...")

    elif key == ord("q"):

        print("\nStopping...")
        break


# -----------------------------------
# Cleanup
# -----------------------------------

cap.release()

file.close()

cv2.destroyAllWindows()

landmarker.close()

print("Dataset collection stopped.")