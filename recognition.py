import face_recognition
import cv2
import os
import numpy as np
import threading
import random

# Step 1: Load known faces and their names
known_face_encodings = []
known_face_names = []
face_colors = {}  # Store unique colors for each recognized person

# Folder where known face images are stored
known_faces_dir = "image_recognition"

# Check if the directory exists
if not os.path.exists(known_faces_dir):
    print(f"Error: Directory '{known_faces_dir}' not found!")
    exit(1)

# Function to generate a unique non-red color
def get_unique_color():
    while True:
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        if color != (0, 0, 255):  # Avoid red
            return color

# Load known faces
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            face_colors[name] = get_unique_color()  # Assign a unique color
        else:
            print(f"Warning: No face found in {filename}")

# Step 2: Initialize Video Capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Try to increase FPS

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

print("Starting video stream. Press 'q' to quit.")

frame_skip = 2  # Skip every 2nd frame for optimization
frame_count = 0
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True  # Process one frame, skip the next

# Function to run face recognition in a separate thread
def recognize_faces(rgb_frame):
    global face_locations, face_encodings, face_names
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)  # Reduce frame size for speed
    
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        color = (0, 0, 255)  # Default color for unknown faces (red)

        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = face_colors.get(name, (0, 255, 0))  # Assign a unique color

        face_names.append((name, color))

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:  # Skip processing every nth frame
            continue

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face recognition in a separate thread
        if process_this_frame:
            thread = threading.Thread(target=recognize_faces, args=(rgb_frame,))
            thread.start()
        
        process_this_frame = not process_this_frame  # Process every other frame

        for (top, right, bottom, left), (name, color) in zip(face_locations, face_names):
            # Scale face location back to the original frame size
            top = int(top / 0.5)
            right = int(right / 0.5)
            bottom = int(bottom / 0.5)
            left = int(left / 0.5)

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
