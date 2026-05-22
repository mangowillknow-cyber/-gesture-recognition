import cv2


def count_extended_fingers(landmarks, hand_label):
    """Count extended fingers for one hand.

    Args:
        landmarks: MediaPipe hand landmarks (21 points)
        hand_label: "Left" or "Right"

    Returns:
        Number of extended fingers (0-5)
    """
    count = 0

    # Thumb: compare x position of tip(4) vs ip(3)
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1

    # Other 4 fingers: tip above pip means extended
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


def classify_gesture(hands_data):
    """Classify gesture from one or two hands.

    Args:
        hands_data: list of (landmarks, hand_label) tuples

    Returns:
        Gesture name string
    """
    if not hands_data:
        return "No hand detected"

    if len(hands_data) == 1:
        lm, label = hands_data[0]
        fingers = count_extended_fingers(lm, label)

        if fingers == 0:
            return "Fist"
        if fingers == 1:
            thumb_up = (lm[4].x < lm[3].x) if label == "Right" else (lm[4].x > lm[3].x)
            index_down = lm[8].y > lm[6].y
            if thumb_up and index_down:
                return "Thumbs Up"
            return "1"
        if fingers == 2:
            return "2"
        if fingers == 3:
            return "3"
        if fingers == 4:
            return "4"
        if fingers == 5:
            thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5
            if thumb_index_dist < 0.05:
                return "OK"
            return "5"

    total = sum(count_extended_fingers(lm, label) for lm, label in hands_data)
    if total == 10:
        return "10"
    if 6 <= total <= 9:
        return str(total)

    return "Unknown"


class CameraManager:
    def __init__(self):
        self.cap = None
        self.current_index = -1

    def enumerate_cameras(self, max_test=10):
        """Detect available cameras by testing indices."""
        cameras = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        return cameras

    def open(self, index):
        """Open camera at given index."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        self.current_index = index
        return self.cap.isOpened()

    def read(self):
        """Read a frame. Returns (success, frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
