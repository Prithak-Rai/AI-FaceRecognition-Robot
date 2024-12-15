import cv2
import face_recognition
import os
import time
import tempfile

# Load Camera
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

capture_delay = 3.0  # Adjust the delay as needed

# Path to the Data directory
data_dir = "Data"
saved_images_dir = "Unknown"  # Custom directory for saving images

# Create the directory if it doesn't exist
if not os.path.exists(saved_images_dir):
    os.makedirs(saved_images_dir)

# Initialize lists to hold face encodings and names
known_face_encodings = []
known_face_names = []

# Load images from subdirectories inside Data
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):  # Check if it's a folder
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            
            # Check if the file is an image based on the extension
            if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Load the image and encode faces
                known_image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(known_image)
                
                if encodings:  # If face encodings are found
                    known_face_encodings.append(encodings[0])  # Take the first encoding found
                    known_face_names.append(person_folder)  # Use the folder name as the person's name

# Continuously capture frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Convert frame from BGR (OpenCV format) to RGB (face_recognition format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for face_loc, face_encoding in zip(face_locations, face_encodings):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Check if the detected face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            b, g, r = 0, 255, 0  # Green for recognized face
        else:
            name = "Unknown"
            b, g, r = 0, 0, 255  # Red for unknown face

            # Save the image of the unknown person to the custom directory
            timestamp = time.strftime("%Y%m%d-%H%M%S")  # Use timestamp as filename
            image_filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join(saved_images_dir, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

        # Display the name and rectangle around the face
        font_scale = 1.5
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
