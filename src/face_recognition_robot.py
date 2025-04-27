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
import serial

# Configuration
DB_PATH = "faces.db"
SAVED_UNKNOWN_DIR = "Unknown"
EMAIL_ADDRESS = "prithak.khamtu@gmail.com"
EMAIL_PASSWORD = "paykcwhdbymsukrk" 
RECEIVER_EMAIL = "prithakhamtu@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SIMILARITY_THRESHOLD = 0.5

# Robot Control Config
SERIAL_PORT = '/dev/cu.usbserial-1120'
SERIAL_BAUDRATE = 115200
DEBUG_MODE = False

# Servo angle ranges
x_min = 0
x_mid = 75
x_max = 150

y_min = 0
y_mid = 90
y_max = 180

# Initialize default servo angles
servo_angles = [x_mid, y_mid]
prev_servo_angles = servo_angles.copy()

def setup_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
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
    logging.info("✅Loaded faces.")
    return conn, cursor, known_embeddings, known_names

def send_email_with_image(cropped_face, subject="Unknown Person Detected"):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText("An unknown person was detected by your security system.", 'plain'))
    
    cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_face_rgb)
    
    temp_path = os.path.join(SAVED_UNKNOWN_DIR, f"temp_cropped_face.jpg")
    pil_image.save(temp_path)
    
    with open(temp_path, 'rb') as f:
        img_data = f.read()
    msg.attach(MIMEImage(img_data, name="unknown_face.jpg"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email sent with cropped face.")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

def map_face_to_servo(face_center, frame_width, frame_height):
    x, y = face_center
    
    x_norm = x / frame_width
    y_norm = y / frame_height
    
    servo_x = int(x_norm * (x_max - x_min) + x_min)
    servo_y = int((1 - y_norm) * (y_max - y_min) + y_min)
    
    servo_x = max(x_min, min(x_max, servo_x))
    servo_y = max(y_min, min(y_max, servo_y))
    
    return [servo_x, servo_y]

def send_servo_command(angles, ser):
    if DEBUG_MODE:
        print(f"DEBUG: Servo angles: {angles}")
    else:
        try:
            ser.write(bytearray(angles))
            print(f"Sent servo angles: {angles}")
        except Exception as e:
            print(f"Error sending servo command: {e}")

def main():
    # Initialize servo angles at global scope
    global servo_angles, prev_servo_angles
    
    # Setup
    device, mtcnn, resnet, preprocess = setup_models()
    conn, cursor, known_embeddings, known_names = load_known_faces(DB_PATH, device, resnet, preprocess)
    
    if not os.path.exists(SAVED_UNKNOWN_DIR):
        os.makedirs(SAVED_UNKNOWN_DIR)

    # Initialize serial connection
    if not DEBUG_MODE:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)
            time.sleep(2)
            print(f"✅ Connected to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
        except Exception as e:
            print(f"❌ Failed to connect to serial port: {e}")
            return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    processed_faces = set()
    prev_time = 0
    frame_rate = 15

    while True:
        current_time = time.time()
        if current_time - prev_time < 1.0 / frame_rate:
            continue
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            faces, probs = mtcnn.detect(rgb_small_frame)

        # Reset servo angles if no face detected
        if faces is None:
            servo_angles = [x_mid, y_mid]
            if servo_angles != prev_servo_angles:
                if not DEBUG_MODE:
                    send_servo_command(servo_angles, ser)
                prev_servo_angles = servo_angles.copy()
            continue

        for i, box in enumerate(faces):
            if probs[i] < 0.90:
                continue
            x1, y1, x2, y2 = [int(coord * 2) for coord in box]
            face_img = frame[y1:y2, x1:x2]

            try:
                face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    face_embedding = resnet(face_tensor).cpu().numpy()

                face_hash = hash(tuple(face_embedding.tobytes()))

                name = "Unknown"
                for emb, known_name in zip(known_embeddings, known_names):
                    similarity = cosine_similarity(face_embedding, emb)
                    distance = 1 - similarity
                    if distance < SIMILARITY_THRESHOLD:
                        name = known_name
                        break

                if name == "Unknown" and face_hash not in processed_faces:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    filename = f"Unknown_{timestamp}.jpg"
                    filepath = os.path.join(SAVED_UNKNOWN_DIR, filename)
                    
                    cropped_face = frame[y1:y2, x1:x2]
                    cv2.imwrite(filepath, cropped_face)
                    
                    if save_unknown_person(cropped_face, face_embedding, cursor):
                        conn.commit()
                        send_email_with_image(cropped_face)
                    
                    processed_faces.add(face_hash)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Face following logic
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                servo_angles = map_face_to_servo(face_center, frame_width, frame_height)

                if servo_angles != prev_servo_angles:
                    if not DEBUG_MODE:
                        send_servo_command(servo_angles, ser)
                    prev_servo_angles = servo_angles.copy()

            except Exception as e:
                print(f"Face processing error: {e}")

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    if not DEBUG_MODE and 'ser' in locals():
        ser.close()

if __name__ == "__main__":
    main()