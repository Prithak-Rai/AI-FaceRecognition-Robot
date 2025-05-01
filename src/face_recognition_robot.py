import sqlite3
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import torchvision.transforms as transforms
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import logging
import threading

DB_PATH = "faces.db"
SAVED_UNKNOWN_DIR = "Unknown"
EMAIL_ADDRESS = "prithak.khamtu@gmail.com"
EMAIL_PASSWORD = "paykcwhdbymsukrk" 
RECEIVER_EMAIL = "prithakhamtu@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SIMILARITY_THRESHOLD = 0.5

def setup_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models with optimized settings for Raspberry Pi
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return device, mtcnn, resnet, preprocess

def load_known_faces(db_path, device, resnet, preprocess):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    known_embeddings = []
    known_names = []

    cursor.execute("""
        SELECT person.id, person.name, faces.image
        FROM person
        JOIN faces ON person.id = faces.person_id
    """)
    rows = cursor.fetchall()

    for person_id, person_name, image_blob in rows:
        try:
            image = Image.open(io.BytesIO(image_blob)).convert("RGB")
            face_tensor = preprocess(np.array(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()
            known_embeddings.append(embedding)
            known_names.append(person_name)
        except Exception as e:
            print(f"Error loading image for {person_name}: {e}")

    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Loaded known faces.")
    return conn, cursor, known_embeddings, known_names

def send_email_async(cropped_face, subject="Unknown Person Detected"):
    def _send():
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = subject
            msg.attach(MIMEText("An unknown person was detected by your security system.", 'plain'))
            
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_face_rgb)
            
            temp_path = os.path.join(SAVED_UNKNOWN_DIR, f"temp_cropped_face_{int(time.time())}.jpg")
            pil_image.save(temp_path)
            
            with open(temp_path, 'rb') as f:
                img_data = f.read()
            msg.attach(MIMEImage(img_data, name="unknown_face.jpg"))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"✉️ Email sent with cropped face.")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    threading.Thread(target=_send, daemon=True).start()

def save_unknown_person(cropped_face, embedding, db_cursor):
    try:
        _, img_encoded = cv2.imencode('.jpg', cropped_face)
        img_bytes = img_encoded.tobytes()
        embedding_bytes = embedding.tobytes()
        
        db_cursor.execute("""
            INSERT INTO unknown_persons (image, embedding)
            VALUES (?, ?)
        """, (img_bytes, embedding_bytes))
        print("✅ Saved unknown person to database")
        return True
    except Exception as e:
        print(f"Error saving unknown person to database: {e}")
        return False

def initialize_camera():
    # Try different camera backends and devices
    for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
        for device in [0, 10, 12]:
            cap = cv2.VideoCapture(device, backend)
            if cap.isOpened():
                # Camera warm-up
                for _ in range(5):
                    cap.read()
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                print(f"✅ Camera initialized on /dev/video{device} with backend {backend}")
                return cap
    
    print("❌ Could not initialize any camera")
    return None

def main():
    # Setup
    device, mtcnn, resnet, preprocess = setup_models()
    conn, cursor, known_embeddings, known_names = load_known_faces(DB_PATH, device, resnet, preprocess)
    
    os.makedirs(SAVED_UNKNOWN_DIR, exist_ok=True)
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        exit()

    processed_faces = set()
    prev_time = time.time()
    fps = 0

    while True:
        # Read frame with validation
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("⚠️ Invalid frame, reinitializing camera...")
            cap.release()
            time.sleep(1)
            cap = initialize_camera()
            if cap is None:
                break
            continue

        # Calculate FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
        prev_time = current_time

        try:
            # Validate frame before processing
            if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                continue

            # Resize for face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            with torch.no_grad():
                faces, probs = mtcnn.detect(rgb_small_frame)

            if faces is not None:
                for i, box in enumerate(faces):
                    if probs[i] < 0.90:
                        continue
                    
                    # Scale coordinates
                    x1, y1, x2, y2 = [int(coord * 2) for coord in box]
                    face_img = frame[y1:y2, x1:x2]

                    try:
                        # Get embedding
                        face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            face_embedding = resnet(face_tensor).cpu().numpy()

                        face_hash = hash(tuple(face_embedding.tobytes()))

                        # Compare with known faces
                        name = "Unknown"
                        for emb, known_name in zip(known_embeddings, known_names):
                            similarity = cosine_similarity(face_embedding, emb)
                            if similarity > SIMILARITY_THRESHOLD:
                                name = known_name
                                break

                        # Handle unknown face
                        if name == "Unknown" and face_hash not in processed_faces:
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            filename = f"Unknown_{timestamp}.jpg"
                            filepath = os.path.join(SAVED_UNKNOWN_DIR, filename)
                            
                            cv2.imwrite(filepath, face_img)
                            
                            if save_unknown_person(face_img, face_embedding, cursor):
                                conn.commit()
                                send_email_async(face_img)
                            
                            processed_faces.add(face_hash)

                        # Draw bounding box
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} ({fps:.1f}fps)", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    except Exception as e:
                        print(f"Face processing error: {e}")

            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Processing error: {e}")
            continue

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()