# Gesture Recognition Software Design

## Overview

A desktop application that recognizes hand gestures (numbers 1-10, fist, thumbs up, OK) using a webcam, including Iriun Webcam for phone cameras.

## Tech Stack

- Python 3.10+
- OpenCV (video capture and display)
- MediaPipe Hands (hand landmark detection)
- PyQt5 (GUI with camera selection dropdown)
- NumPy

## Architecture

Single file `gesture_recognition.py` with these logical sections:

1. **Camera Manager** — Enumerates system cameras (including Iriun), provides dropdown selection, handles camera switching
2. **Gesture Detector** — Uses MediaPipe Hands to detect hand landmarks, applies rules to classify gestures
3. **GUI** — PyQt5 window with dropdown menu, status display, and OpenCV video feed

## Gesture Rules

Based on MediaPipe's 21 hand landmark points, determine finger extension state by comparing fingertip position to PIP joint position.

### Single-hand gestures (1-5)

| Gesture | Rule |
|---------|------|
| 1 | Index finger extended, others curled |
| 2 | Index + middle fingers extended |
| 3 | Index + middle + ring fingers extended |
| 4 | Four fingers extended, thumb curled |
| 5 | All five fingers extended |
| Fist | All fingers curled |
| Thumbs up | Thumb extended, others curled |
| OK | Thumb and index form circle, others extended |

### Two-hand gestures (6-10)

MediaPipe detects up to 2 hands simultaneously. Count extended fingers on each hand and sum.

| Gesture | Rule |
|---------|------|
| 6 | One hand: 1 finger + Other hand: 5 fingers |
| 7 | One hand: 2 fingers + Other hand: 5 fingers |
| 8 | One hand: 3 fingers + Other hand: 5 fingers |
| 9 | One hand: 4 fingers + Other hand: 5 fingers |
| 10 | Both hands: 5 fingers each |

## Camera Support

- Auto-detect all available camera devices via `cv2.VideoCapture`
- Iriun Webcam appears as a standard video capture device — no special handling needed
- Dropdown menu lists all detected cameras by index/name
- Switching cameras releases current capture and opens selected device

## UI Layout

```
+------------------------------------------+
| [Camera: Dropdown v]  Status: Ready      |
| +--------------------------------------+ |
| |                                      | |
| |        OpenCV Video Feed             | |
| |    (gesture overlay drawn here)      | |
| |                                      | |
| +--------------------------------------+ |
| Detected: [gesture name]                |
+------------------------------------------+
```

## Performance Considerations

- MediaPipe runs at ~30 FPS on CPU
- Process every frame (no skipping) for responsive detection
- Smooth gesture output with 3-frame voting to reduce flicker

## Dependencies

```
opencv-python>=4.8
mediapipe>=0.10
pyqt5>=5.15
numpy>=1.24
```

## File Structure

```
E:\Programming\Claude\gesture_recognition.py   # Single file application
```
