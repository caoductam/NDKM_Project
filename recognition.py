import face_recognition
import cv2
import os
import numpy as np

# Step 1: Load known faces and their names
known_face_encodings = []
known_face_names = []

# Folder where known face images are stored
known_faces_dir = "image_recognition"

# Check if the directory exists
if not os.path.exists(known_faces_dir):
    print(f"Error: Directory '{known_faces_dir}' not found!")
    exit(1)

# Loop through each file in the known_faces directory
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"Warning: No face found in {filename}")

# Step 2: Initialize Video Capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

print("Starting video stream. Press 'q' to quit.")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 3: Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            color = (0, 0, 255)  # Default to red (unrecognized)

            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for recognized faces

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Draw a label with the name
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Video', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Step 5: Clean up
    video_capture.release()
    cv2.destroyAllWindows()
