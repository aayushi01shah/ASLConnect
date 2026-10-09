import os
from collections import Counter, deque

from letterFeatures import extractFeatures

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_model.joblib")
MIN_CONFIDENCE = 0.6   # ignore letter guesses below this
SMOOTHING_FRAMES = 7   # a letter must win a vote over this many recent frames

_model = None
_modelChecked = False
_recent = deque(maxlen=SMOOTHING_FRAMES)


def _getModel():
    global _model, _modelChecked
    if not _modelChecked:
        _modelChecked = True
        if os.path.exists(MODEL_PATH):
            import joblib
            _model = joblib.load(MODEL_PATH)
        else:
            print("No letter_model.joblib found - only 'I love you' will be detected. "
                  "Run collect_data.py then train.py to enable letters.")
    return _model


def _predictLetter(results):
    """Return the letter for the current frame, or None."""
    model = _getModel()
    if model is None or not results.hand_landmarks:
        return None
    feats = extractFeatures(results.hand_landmarks[0], results.handedness[0][0].category_name)
    if feats is None:
        return None
    probs = model.predict_proba([feats])[0]
    best = probs.argmax()
    return str(model.classes_[best]) if probs[best] >= MIN_CONFIDENCE else None


def processResults(mpImg, recognizer):
    recognitionResults = recognizer.recognize(mpImg)

    label = "None"

    if len(recognitionResults.gestures) != 0:
        topGesture = recognitionResults.gestures[0][0]
        if topGesture.category_name == "ILoveYou" and topGesture.score >= 0.5:
            _recent.clear()
            return recognitionResults, "I love you"

    if not recognitionResults.hand_landmarks:
        _recent.clear()
        return recognitionResults, label

    _recent.append(_predictLetter(recognitionResults))
    letter, votes = Counter(_recent).most_common(1)[0]
    if letter is not None and votes > len(_recent) // 2:
        label = letter

    return recognitionResults, label
