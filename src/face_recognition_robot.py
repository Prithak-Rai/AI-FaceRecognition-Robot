import cv2
import face_recognition
import os
import time

# Load Camera
cap = None
for i in range(4):
    # cap = cv2.VideoCapture("http://10.22.19.228:8080/video")
    # cap = cv2.VideoCapture("http://192.168.0.100:8080/video")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

# Reduce resolution for smoother streaming
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  

capture_delay = 1.5  # Adjust the delay as needed

# Path to the Data directory
data_dir = "Data"
saved_images_dir = "Unknown"

if not os.path.exists(saved_images_dir):
    os.makedirs(saved_images_dir)

known_face_encodings = []
known_face_names = []

for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            
            # Check if the file is an image based on the extension
            if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                #Load the image and encode faces
                known_image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(known_image)

                if encodings:
                    known_face_encodings.append(encodings[0]) # Take the first encoding found
                    known_face_names.append(person_folder) # Use the folder name as the person's name

prev_time = time.time()
frame_count = 0  # For frame skipping

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    prev_time = current_time

    frame_count += 1

    # Skip every alternate frame to improve FPS
    if frame_count % 2 == 0:
        continue

    # Downscale frame for face detection to increase processing speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb, model='hog')
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for face_loc, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_loc
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join(saved_images_dir, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
