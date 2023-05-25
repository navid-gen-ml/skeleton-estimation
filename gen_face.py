import cv2
import mediapipe as mp
import numpy as np

# Set up MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
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

    # Run face mesh model on frame
    results = face_mesh.process(frame_rgb)

    # Extract face landmarks and add to output data
    if results.multi_face_landmarks:
        face_landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            face_landmarks.append([landmark.x, landmark.y, landmark.z])
        output_data.append(np.array(face_landmarks))

    # Show video frame with face landmarks
    if results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_LIPS,
            mp_drawing.DrawingSpec(color=(0, 255, 255),
                                   thickness=1,
                                   circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 0),
                                   thickness=1,
                                   circle_radius=1),
        )
    cv2.imshow("FaceMesh", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Save output data to numpy array
output_data = np.array(output_data)
np.save("file_face-land.npy", output_data)
