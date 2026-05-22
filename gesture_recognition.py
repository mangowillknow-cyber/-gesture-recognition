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
