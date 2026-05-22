# Gesture Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyQt5 desktop app that recognizes hand gestures (numbers 1-10, fist, thumbs up, OK) from webcam input including Iriun Webcam.

**Architecture:** Single Python file with camera management, MediaPipe hand detection, gesture classification logic, and PyQt5 GUI. Gesture logic is tested separately via unit tests.

**Tech Stack:** Python 3.10+, OpenCV, MediaPipe Hands, PyQt5, NumPy

---

## File Structure

```
E:\Programming\Claude\
  gesture_recognition.py          # Main application (single file)
  requirements.txt                # Dependencies
  tests/
    test_gesture_logic.py         # Unit tests for gesture classification
```

---

### Task 1: Project Setup

**Files:**
- Create: `E:\Programming\Claude\requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
opencv-python>=4.8
mediapipe>=0.10
pyqt5>=5.15
numpy>=1.24
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r "E:/Programming/Claude/requirements.txt"`

- [ ] **Step 3: Commit**

```bash
cd "E:/Programming/Claude"
git add requirements.txt
git commit -m "feat: add project dependencies"
```

---

### Task 2: Gesture Logic — Finger Counting

**Files:**
- Create: `E:\Programming\Claude\tests\test_gesture_logic.py`
- Create: `E:\Programming\Claude\gesture_recognition.py`

- [ ] **Step 1: Write the failing test for finger counting**

Create `E:\Programming\Claude\tests\test_gesture_logic.py`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_recognition import count_extended_fingers


def test_thumb_extended():
    """Thumb extended when tip.x < ip.x for right hand (mirrored)."""
    landmarks = [0] * 21
    # Thumb: tip(4) left of ip(3) for right hand
    landmarks[4] = type('LM', (), {'x': 0.3, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    # Other fingers: tip above pip = extended
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.3})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    assert count_extended_fingers(landmarks, "Right") == 5


def test_thumb_curled():
    """Thumb curled when tip.x > ip.x for right hand."""
    landmarks = [0] * 21
    landmarks[4] = type('LM', (), {'x': 0.6, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.6})()
    assert count_extended_fingers(landmarks, "Right") == 0


def test_index_only():
    """Only index extended = 1 finger."""
    landmarks = [0] * 21
    landmarks[4] = type('LM', (), {'x': 0.6, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    # Index extended
    landmarks[8] = type('LM', (), {'x': 0.5, 'y': 0.3})()
    landmarks[6] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    # Others curled
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.6})()
    assert count_extended_fingers(landmarks, "Right") == 1


def test_left_hand_thumb():
    """Left hand thumb logic is mirrored."""
    landmarks = [0] * 21
    landmarks[4] = type('LM', (), {'x': 0.7, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.3})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    assert count_extended_fingers(landmarks, "Left") == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "E:/Programming/Claude" && python -m pytest tests/test_gesture_logic.py -v`
Expected: FAIL with `ImportError: cannot import name 'count_extended_fingers'`

- [ ] **Step 3: Implement count_extended_fingers**

Create `E:\Programming\Claude\gesture_recognition.py`:

```python
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys


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
    # MediaPipe mirrors the image, so for right hand in image it appears as left
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "E:/Programming/Claude" && python -m pytest tests/test_gesture_logic.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py tests/test_gesture_logic.py
git commit -m "feat: add finger counting logic with tests"
```

---

### Task 3: Gesture Logic — Gesture Classification

**Files:**
- Modify: `E:\Programming\Claude\gesture_recognition.py`
- Modify: `E:\Programming\Claude\tests\test_gesture_logic.py`

- [ ] **Step 1: Write failing test for classify_gesture**

Append to `E:\Programming\Claude\tests\test_gesture_logic.py`:

```python
from gesture_recognition import classify_gesture


def _make_landmarks(finger_states):
    """Helper: finger_states = [thumb, index, middle, ring, pinky] as bools."""
    landmarks = [None] * 21
    # Thumb
    landmarks[4] = type('LM', (), {'x': 0.3 if finger_states[0] else 0.7, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    # Other fingers: extended = tip.y < pip.y
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
        extended = finger_states[i + 1]
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.3 if extended else 0.5})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    return landmarks


def test_single_hand_fist():
    lm = _make_landmarks([False, False, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "Fist"


def test_single_hand_one():
    lm = _make_landmarks([False, True, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "1"


def test_single_hand_five():
    lm = _make_landmarks([True, True, True, True, True])
    assert classify_gesture([(lm, "Right")]) == "5"


def test_single_hand_thumbs_up():
    lm = _make_landmarks([True, False, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "Thumbs Up"


def test_two_hands_six():
    lm1 = _make_landmarks([False, True, False, False, False])   # 1 finger
    lm2 = _make_landmarks([True, True, True, True, True])       # 5 fingers
    assert classify_gesture([(lm1, "Right"), (lm2, "Left")]) == "6"


def test_two_hands_ten():
    lm1 = _make_landmarks([True, True, True, True, True])  # 5
    lm2 = _make_landmarks([True, True, True, True, True])  # 5
    assert classify_gesture([(lm1, "Right"), (lm2, "Left")]) == "10"


def test_no_hands():
    assert classify_gesture([]) == "No hand detected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "E:/Programming/Claude" && python -m pytest tests/test_gesture_logic.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_gesture'`

- [ ] **Step 3: Implement classify_gesture**

Append to `E:\Programming\Claude\gesture_recognition.py`:

```python
def classify_gesture(hands_data):
    """Classify gesture from one or two hands.

    Args:
        hands_data: list of (landmarks, hand_label) tuples

    Returns:
        Gesture name string
    """
    if not hands_data:
        return "No hand detected"

    # Single hand
    if len(hands_data) == 1:
        lm, label = hands_data[0]
        fingers = count_extended_fingers(lm, label)

        if fingers == 0:
            return "Fist"
        if fingers == 1:
            # Check if it's thumbs up (only thumb extended)
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
            # Check OK gesture: thumb and index close together
            thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5
            if thumb_index_dist < 0.05:
                return "OK"
            return "5"

    # Two hands: sum fingers
    total = sum(count_extended_fingers(lm, label) for lm, label in hands_data)
    if total == 10:
        return "10"
    if 6 <= total <= 9:
        return str(total)

    return "Unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "E:/Programming/Claude" && python -m pytest tests/test_gesture_logic.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py tests/test_gesture_logic.py
git commit -m "feat: add gesture classification with tests"
```

---

### Task 4: Camera Manager

**Files:**
- Modify: `E:\Programming\Claude\gesture_recognition.py`

- [ ] **Step 1: Add camera enumeration and switching**

Append to `E:\Programming\Claude\gesture_recognition.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py
git commit -m "feat: add camera manager"
```

---

### Task 5: MediaPipe Integration

**Files:**
- Modify: `E:\Programming\Claude\gesture_recognition.py`

- [ ] **Step 1: Add hand detector class**

Append to `E:\Programming\Claude\gesture_recognition.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py
git commit -m "feat: add MediaPipe hand detector"
```

---

### Task 6: Gesture Smoothing

**Files:**
- Modify: `E:\Programming\Claude\gesture_recognition.py`

- [ ] **Step 1: Add gesture smoother class**

Append to `E:\Programming\Claude\gesture_recognition.py`:

```python
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

        # Return most common gesture in window
        from collections import Counter
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.history.clear()
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py
git commit -m "feat: add gesture smoother for stable output"
```

---

### Task 7: PyQt5 GUI

**Files:**
- Modify: `E:\Programming\Claude\gesture_recognition.py`

- [ ] **Step 1: Add main window class**

Replace the entire content of `E:\Programming\Claude\gesture_recognition.py` with:

```python
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
from collections import Counter


def count_extended_fingers(landmarks, hand_label):
    """Count extended fingers for one hand."""
    count = 0
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count


def classify_gesture(hands_data):
    """Classify gesture from one or two hands."""
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
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        self.current_index = index
        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
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
    def __init__(self, window=3):
        self.history = []
        self.window = window

    def update(self, gesture):
        self.history.append(gesture)
        if len(self.history) > self.window:
            self.history.pop(0)
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.history.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition")
        self.setMinimumSize(800, 600)

        self.camera = CameraManager()
        self.detector = HandDetector()
        self.smoother = GestureSmoother()

        self._setup_ui()
        self._init_camera()

        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(30)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top bar: camera selection
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_change)
        top.addWidget(self.camera_combo)
        self.status_label = QLabel("Status: Initializing...")
        top.addWidget(self.status_label)
        top.addStretch()
        layout.addLayout(top)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1a1a2e;")
        layout.addWidget(self.video_label)

        # Bottom: detected gesture
        self.gesture_label = QLabel("Detected: ---")
        self.gesture_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.gesture_label)

    def _init_camera(self):
        cameras = self.camera.enumerate_cameras()
        if not cameras:
            self.status_label.setText("Status: No camera found")
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)
        self._open_camera(cameras[0])

    def _on_camera_change(self, index):
        if index >= 0:
            cam_id = self.camera_combo.itemData(index)
            self._open_camera(cam_id)

    def _open_camera(self, cam_id):
        self.camera.release()
        self.smoother.reset()
        if self.camera.open(cam_id):
            self.status_label.setText(f"Status: Camera {cam_id} active")
        else:
            self.status_label.setText(f"Status: Failed to open Camera {cam_id}")

    def _process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        hands_data, frame = self.detector.detect(frame)
        gesture = classify_gesture(hands_data)
        smoothed = self.smoother.update(gesture)

        self.gesture_label.setText(f"Detected: {smoothed}")

        # Convert frame to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.timer.stop()
        self.detector.release()
        self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Programming/Claude"
git add gesture_recognition.py
git commit -m "feat: complete gesture recognition app with PyQt5 GUI"
```

---

### Task 8: Final Testing

- [ ] **Step 1: Run all unit tests**

Run: `cd "E:/Programming/Claude" && python -m pytest tests/test_gesture_logic.py -v`
Expected: All 11 tests PASS

- [ ] **Step 2: Run the application**

Run: `cd "E:/Programming/Claude" && python gesture_recognition.py`
Expected: Window opens, camera feed displays, gesture detection works

- [ ] **Step 3: Test camera switching**

- Select different cameras from dropdown
- Verify Iriun Webcam appears in the list
- Verify video feed updates when switching

- [ ] **Step 4: Test gesture recognition**

- Show numbers 1-5 with one hand
- Show numbers 6-10 with two hands
- Show fist, thumbs up, OK gestures
- Verify smooth output (no flickering)

- [ ] **Step 5: Commit final state**

```bash
cd "E:/Programming/Claude"
git add -A
git commit -m "feat: gesture recognition app complete"
```
