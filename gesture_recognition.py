import cv2
import mediapipe as mp


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


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """Detect hands in frame.

        Returns:
            hands_data: list of (landmarks, label) tuples
            frame_with_drawing: frame with landmarks drawn
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_data = []
        if results.multi_hand_landmarks:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                hands_data.append((hand_lm.landmark, label))
                self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

        return hands_data, frame

    def release(self):
        self.hands.close()


class GestureSmoother:
    """Smooth gesture output using voting over recent frames."""

    def __init__(self, window=3):
        self.history = []
        self.window = window

    def update(self, gesture):
        """Add new gesture and return smoothed result."""
        self.history.append(gesture)
        if len(self.history) > self.window:
            self.history.pop(0)

        from collections import Counter
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.history.clear()
