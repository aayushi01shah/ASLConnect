import numpy as np

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"  # static ASL letters (J and Z need motion)


def extractFeatures(handLandmarks, handednessName):
    """Turn 21 MediaPipe hand landmarks into a 63-number feature vector.

    The vector doesn't depend on where the hand is in the frame, how big it
    is, or which hand it is, so the same classifier works for left and right.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks], dtype=np.float32)
    pts -= pts[0]                      # wrist becomes the origin
    if handednessName == "Left":       # mirror left hands onto right hands
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale < 1e-6:
        return None
    return (pts / scale).flatten()
