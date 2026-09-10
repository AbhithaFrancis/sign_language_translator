# sign_language_translator

A real-time sign language recognition system that uses a webcam, MediaPipe hand-landmark detection, and a Random Forest machine learning model to recognize predefined hand signs and convert them into text and speech.

The system detects 21 hand landmarks (63 coordinates) from the user's hand, normalizes the landmark data to reduce the effect of hand position and size, and uses the trained Random Forest classifier to identify the sign. A 90% confidence threshold is used to reduce uncertain predictions. Recognized signs are displayed on the screen and converted into speech using Windows Text-to-Speech.

Features:-
Real-time hand detection using MediaPipe
21-point hand landmark extraction
Landmark normalization using wrist position and hand size
Random Forest-based sign classification
90% confidence threshold
Real-time webcam recognition
Text-to-speech output
Custom dataset collection and model training

Technologies Used:-
Python,
OpenCV,
MediaPipe,
NumPy,
Pandas,
Scikit-learn,
Joblib,
Windows System.Speech.

Current Sign Classes:-
HELLO,
YES,
NO,
THANK YOU,
STOP,
LOVE,
FRIEND.
