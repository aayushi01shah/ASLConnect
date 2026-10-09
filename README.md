# ASLConnect
This project is a Sign Language to English Translator, which utilizes a machine learning model for hand tracking and gesture recognition. The translation is done instantly on-device, ensuring quick processing and detection of gestures.


# Features
Converts sign language gestures into English text.
Utilizes machine learning for hand tracking and gesture recognition.
Instant processing and detection of gestures on-device.
Technologies Used
Python
Mediapipe library
OpenCV
# How to Use
Ensure you have Python installed on your system.
Install necessary dependencies:
pip install mediapipe
pip install numpy
pip install cv2
Clone this repository to your local machine.
Run the main Python script:
python main.py
The English Translation will pop up in the top right corner.
Currently only the phrase 'I Love You' or 🤟 is working
Press Q to quit.
# Acknowledgments
This project was submitted to De Anza Hacks, where it won Best Educational Hack from a pool of 30+ submissions.

# Contributors
Aayushi Shah and Aayush Shah


# ASL Letters (A-Y)

The built-in MediaPipe model only knows a handful of generic gestures, so letters use a small classifier trained on your own hand landmarks.

1. Install: `pip install -r requirements.txt`
2. Record samples: `python collect_data.py`. Press a letter key (A-Y, no J) while making that sign. It records 60 frames per press, so tilt and shift your hand slowly as it records. Do 2-3 bursts per letter, ideally in different lighting and distances. ESC quits.
3. Train: `python train.py` (prints held-out accuracy, saves `letter_model.joblib`)
4. Run: `python main.py`. The detected letter shows on screen; "I love you" still works.

J and Z involve motion and aren't supported. Look-alike signs (M/N/S/T/A/E) need more samples to separate.
