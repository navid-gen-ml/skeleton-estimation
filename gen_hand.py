import cv2
import mediapipe as mp
import numpy as np

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# Set up MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils

# Read video file
cap = cv2.VideoCapture(
    "/media/shahidul/store1/SD17-V-A10/npy/rnd/Abdullah_1.mp4")

# Define output numpy array
output_data = []

while cap.isOpened():
    # Read frame from video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run hands model on frame
    results = hands.process(frame_rgb)

    # Extract hand landmarks and add to output data
    hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand_idx in range(len(results.multi_hand_landmarks)):
            for landmark in results.multi_hand_landmarks[hand_idx].landmark:
                hand_landmarks.append([landmark.x, landmark.y, landmark.z])
    if len(hand_landmarks) > 0:
        output_data.append(np.array(hand_landmarks))
    else:
        # Add padding if no hand landmarks detected in this frame
        output_data.append(np.zeros((21 * 2, 3)))

    # Show video frame with hand landmarks
    if results.multi_hand_landmarks:
        for hand_idx in range(len(results.multi_hand_landmarks)):
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[hand_idx],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255),
                                       thickness=2,
                                       circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255),
                                       thickness=2,
                                       circle_radius=2),
            )
    cv2.imshow("Hands", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Save output data to numpy array with padding
max_len = max([len(data) for data in output_data])
padded_data = np.zeros((len(output_data), max_len, 3))
for i in range(len(output_data)):
    padded_data[i][:len(output_data[i])] = output_data[i]
np.save("file_hand-land.npy", padded_data)
