def processResults(mpImg, recognizer):
    recognitionResults = recognizer.recognize(mpImg)
    
    label = "None"

    if len(recognitionResults.gestures) != 0:
        topGesture = recognitionResults.gestures[0][0]
        gestureLabel = topGesture.category_name
        if gestureLabel == "ILoveYou" and topGesture.score >= 0.5:
            label = "I love you"
    
    return recognitionResults, label
