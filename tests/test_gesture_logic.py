import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_recognition import count_extended_fingers


def test_thumb_extended():
    """Thumb extended when tip.x < ip.x for right hand (mirrored)."""
    landmarks = [0] * 21
    landmarks[4] = type('LM', (), {'x': 0.3, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
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
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.7})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    assert count_extended_fingers(landmarks, "Right") == 0


def test_index_only():
    """Only index extended = 1 finger."""
    landmarks = [0] * 21
    landmarks[4] = type('LM', (), {'x': 0.6, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    landmarks[8] = type('LM', (), {'x': 0.5, 'y': 0.3})()
    landmarks[6] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.7})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
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


from gesture_recognition import classify_gesture


def _make_landmarks(finger_states, hand_label="Right"):
    """Helper: finger_states = [thumb, index, middle, ring, pinky] as bools."""
    landmarks = [None] * 21
    # Thumb: Right hand uses x < ip.x, Left hand uses x > ip.x
    if hand_label == "Right":
        thumb_x = 0.3 if finger_states[0] else 0.7
    else:
        thumb_x = 0.7 if finger_states[0] else 0.3
    landmarks[4] = type('LM', (), {'x': thumb_x, 'y': 0.5})()
    landmarks[3] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
        extended = finger_states[i + 1]
        landmarks[tip] = type('LM', (), {'x': 0.5, 'y': 0.3 if extended else 0.7})()
        landmarks[pip] = type('LM', (), {'x': 0.5, 'y': 0.5})()
    return landmarks


def test_single_hand_fist():
    lm = _make_landmarks([False, False, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "握拳"


def test_single_hand_one():
    lm = _make_landmarks([False, True, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "1"


def test_single_hand_five():
    lm = _make_landmarks([True, True, True, True, True])
    assert classify_gesture([(lm, "Right")]) == "5"


def test_single_hand_thumbs_up():
    lm = _make_landmarks([True, False, False, False, False])
    assert classify_gesture([(lm, "Right")]) == "点赞"


def test_two_hands_six():
    lm1 = _make_landmarks([False, True, False, False, False], "Right")
    lm2 = _make_landmarks([True, True, True, True, True], "Left")
    assert classify_gesture([(lm1, "Right"), (lm2, "Left")]) == "6"


def test_two_hands_ten():
    lm1 = _make_landmarks([True, True, True, True, True], "Right")
    lm2 = _make_landmarks([True, True, True, True, True], "Left")
    assert classify_gesture([(lm1, "Right"), (lm2, "Left")]) == "10"


def test_two_hands_five():
    lm1 = _make_landmarks([False, True, True, False, False], "Right")  # 2
    lm2 = _make_landmarks([False, True, True, True, False], "Left")    # 3
    assert classify_gesture([(lm1, "Right"), (lm2, "Left")]) == "5"


def test_no_hands():
    assert classify_gesture([]) == "未检测到手势"
