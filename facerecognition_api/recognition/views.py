from django.http import JsonResponse
from django.shortcuts import render
import cv2
import torch
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torchvision.transforms as transforms
import time
from django.conf import settings
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
import base64  # Import base64 module
from django.core.files.storage import FileSystemStorage
import shutil 
from PIL import Image, UnidentifiedImageError

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load known faces from the Data directory outside of the Django app
known_face_embeddings = []
known_face_names = []

def home(request):
    return HttpResponse("Welcome to the Face Recognition API!")

def load_known_faces():
    data_dir = settings.DATA_DIR  # Ensure DATA_DIR is correctly set
    global known_face_embeddings, known_face_names

    known_face_embeddings = []
    known_face_names = []

    for person_folder in os.listdir(data_dir):
        person_folder_path = os.path.join(data_dir, person_folder)
        if os.path.isdir(person_folder_path):
            known_face_names.append(person_folder)
            for image_file in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_file)
                try:
                    
                    image = Image.open(image_path)
                    image = preprocess(image).unsqueeze(0).to(device)
                    embedding = resnet(image).detach().cpu().numpy()
                    known_face_embeddings.append(embedding)
                except (UnidentifiedImageError, IOError):
                    # Skip non-image files
                    continue

# Initialize known faces on server start
load_known_faces()

# Face recognition view
def recognize_face(request):
    cap = cv2.VideoCapture(0)  # Use the webcam (0 is the default webcam)

    if not cap.isOpened():
        return JsonResponse({"error": "Could not open webcam"}, status=400)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return JsonResponse({"error": "Failed to read frame from webcam"}, status=400)

    # Process the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, _ = mtcnn.detect(rgb_frame)

    if faces is None:
        cap.release()
        return JsonResponse({"message": "No face detected"}, status=200)

    # Loop over the detected faces and compare with known faces
    recognized_faces = []
    for i, box in enumerate(faces):
        x1, y1, x2, y2 = map(int, box)
        face_image = rgb_frame[y1:y2, x1:x2]
        face_tensor = preprocess(face_image).unsqueeze(0).to(device)
        face_embedding = resnet(face_tensor).detach().cpu().numpy()

        name = "Unknown"
        color = (0, 0, 255) 
        min_dist = float("inf")

        for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
            dist = np.linalg.norm(face_embedding - known_embedding)
            if dist < 0.8 and dist < min_dist: 
                min_dist = dist
                name = known_name
                color = (0, 255, 0)  # Green for known faces

        if name == "Unknown":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for snap in range(1, 4):
                snap_filename = f"unknown_{timestamp}_{snap}.jpg"
                snap_path = os.path.join(settings.UNK_DIR, snap_filename)
                # Save each snapshot
                cv2.imwrite(snap_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        recognized_faces.append({"name": name, "coordinates": [x1, y1, x2, y2]})

    ret, jpeg_frame = cv2.imencode('.jpg', frame)
    if not ret:
        cap.release()
        return JsonResponse({"error": "Failed to encode frame to JPEG"}, status=400)

    # Return the recognized faces along with the image in response
    img_data = jpeg_frame.tobytes()
    img_data_b64 = base64.b64encode(img_data).decode('utf-8')

    cap.release()
    return JsonResponse({
        "recognized_faces": recognized_faces,
        "image": img_data_b64
    }, status=200)

# View to render the template with the webcam feed
def render_recognition_page(request):
    return render(request, 'recognition/recognize_face.html')

def detect_face(request):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            return JsonResponse({'message': 'Face detected'})
        else:
            return JsonResponse({'message': 'No face detected'})

    cap.release()
    return JsonResponse({'message': 'Failed to capture frame'})

def list_known_faces(request):
    # Return the list of known face names
    return JsonResponse({"known_faces": known_face_names}, status=200)

def add_face(request):
    if request.method == 'POST' and 'name' in request.POST and 'face_image' in request.FILES:
        name = request.POST['name']
        face_image = request.FILES['face_image']

        # Use the DATA_DIR path from settings
        person_folder_path = os.path.join(settings.DATA_DIR, name)

        # Check if folder exists for the person
        if os.path.exists(person_folder_path):
            # Folder exists, update the image in the folder
            filename = face_image.name
            file_path = os.path.join(person_folder_path, filename)

            # Save the image to the folder
            with open(file_path, 'wb') as f:
                for chunk in face_image.chunks():
                    f.write(chunk)

            # Now read the image and process it
            image = cv2.imread(file_path)
            if image is None:
                return render(request, 'recognition/add_face.html', {"error": "Failed to read the image. Please try again."})

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, _ = mtcnn.detect(rgb_image)

            if faces is not None:
                for box in faces:
                    x1, y1, x2, y2 = map(int, box)
                    face_image = rgb_image[y1:y2, x1:x2]
                    face_tensor = preprocess(face_image).unsqueeze(0).to(device)
                    embedding = resnet(face_tensor).detach().cpu().numpy()
                    known_face_embeddings.append(embedding)
                    known_face_names.append(name)

            return render(request, 'recognition/add_face.html', {"message": f"Face for {name} updated successfully."})
        else:
            # Folder doesn't exist, create a new folder and save the image inside
            os.makedirs(person_folder_path)

            # Save the image to the new folder
            filename = face_image.name
            file_path = os.path.join(person_folder_path, filename)

            with open(file_path, 'wb') as f:
                for chunk in face_image.chunks():
                    f.write(chunk)

            # Now read the image and process it
            image = cv2.imread(file_path)
            if image is None:
                return render(request, 'recognition/add_face.html', {"error": "Failed to read the image. Please try again."})

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, _ = mtcnn.detect(rgb_image)

            if faces is not None:
                for box in faces:
                    x1, y1, x2, y2 = map(int, box)
                    face_image = rgb_image[y1:y2, x1:x2]
                    face_tensor = preprocess(face_image).unsqueeze(0).to(device)
                    embedding = resnet(face_tensor).detach().cpu().numpy()
                    known_face_embeddings.append(embedding)
                    known_face_names.append(name)

            return render(request, 'recognition/add_face.html', {"message": f"Face for {name} added successfully."})

    return render(request, 'recognition/add_face.html') 

def delete_face(request):
    if request.method == 'POST' and 'name' in request.POST:
        name = request.POST['name']

        # Check if the folder with the name exists in the DATA_DIR
        person_folder_path = os.path.join(settings.DATA_DIR, name)

        if os.path.isdir(person_folder_path):
            # If the folder exists, delete it
            shutil.rmtree(person_folder_path)  # Removes the folder and all its contents
            
            # Also remove the name from the known_face_names and known_face_embeddings lists
            if name in known_face_names:
                index = known_face_names.index(name)
                del known_face_names[index]
                del known_face_embeddings[index]

            return render(request, 'recognition/delete_face.html', {"message": f"Face {name} and its folder deleted successfully."})
        else:
            return render(request, 'recognition/delete_face.html', {"error": "Folder not found."})

    return render(request, 'recognition/delete_face.html')  # Initial render when no name is provided