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
