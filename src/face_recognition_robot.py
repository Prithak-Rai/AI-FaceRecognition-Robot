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
import serial
import paho.mqtt.client as mqtt
import json
import base64


# Configuration
DB_PATH = "faces.db"
SAVED_UNKNOWN_DIR = "Unknown"
EMAIL_ADDRESS = "prithak.khamtu@gmail.com"
EMAIL_PASSWORD = "paykcwhdbymsukrk" 
RECEIVER_EMAIL = "prithakhamtu@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SIMILARITY_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.3  # For smoother Z-axis movements

# Robot Control Config
SERIAL_PORT = '/dev/cu.usbserial-1120'
SERIAL_BAUDRATE = 115200
DEBUG_MODE = False

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"  # Public broker for testing
MQTT_PORT = 1883
MQTT_TOPIC_NOTIFICATIONS = "facebot/notifications"
MQTT_TOPIC_UNKNOWN_FACE = "facebot/unknown_face"
MQTT_CLIENT_ID = "facebot_raspberry_pi"
MQTT_TOPIC_COMMAND = "facebot/command"  # Topic for incoming commands
MQTT_TOPIC_STATUS = "facebot/status"    # Topic for status updates
MQTT_TOPIC_ROBOT_CONTROL = "facebot/robot/control"  # New topic for robot control
MQTT_TOPIC_CAMERA_SETTINGS = "facebot/camera/settings"  # New topic for camera settings
MQTT_TOPIC_SYSTEM_SETTINGS = "facebot/system/settings"  # New topic for system settings

# Servo angle ranges
x_min = 0
x_mid = 75
x_max = 150

y_min = 0
y_mid = 90
y_max = 180

z_min = 80  # Forward position (face far)
z_max = 220  # Back position (face close)

# Initialize default servo angles
servo_angles = [x_mid, y_mid, 130, 0]  # x, y, z, claw
prev_servo_angles = servo_angles.copy()

# MQTT Client
mqtt_client = None

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to all command topics
    client.subscribe(MQTT_TOPIC_COMMAND)
    client.subscribe(MQTT_TOPIC_ROBOT_CONTROL)
    client.subscribe(MQTT_TOPIC_CAMERA_SETTINGS)
    client.subscribe(MQTT_TOPIC_SYSTEM_SETTINGS)
    client.subscribe("facebot/faces/request", qos=1)  # Add QoS 1 for better reliability
    print("Subscribed to all topics")
    # Publish online status
    client.publish(MQTT_TOPIC_STATUS, json.dumps({
        "type": "robot_status",
        "status": "online",
        "title": "FaceBot Online",
        "body": "Your FaceBot system is now online and monitoring",
        "timestamp": datetime.now().isoformat()
    }))

def on_message(client, userdata, msg):
    print(f"Received MQTT message on topic: {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Message payload: {payload}")
        
        if msg.topic == "facebot/faces/request":
            print("Received faces list request")
            # Add a small delay to ensure MQTT connection is stable
            time.sleep(0.1)
            handle_faces_request()
            
        elif msg.topic == "facebot/faces/command":
            if "command" in payload and payload["command"] == "delete_face":
                if "face_id" in payload:
                    handle_delete_face(payload["face_id"])
            
        elif msg.topic == MQTT_TOPIC_COMMAND:
            if "command" in payload:
                cmd = payload["command"]
                if cmd == "restart":
                    print("Received restart command")
                elif cmd == "shutdown":
                    print("Received shutdown command")
                elif cmd == "status":
                    send_status_update()
                    
        elif msg.topic == MQTT_TOPIC_ROBOT_CONTROL:
            if "action" in payload:
                action = payload["action"]
                if action == "move":
                    if "angles" in payload:
                        angles = payload["angles"]
                        if len(angles) == 4:  # [x, y, z, claw]
                            if not DEBUG_MODE and 'ser' in globals():
                                send_servo_command(angles, ser)
                                print(f"Robot moved to angles: {angles}")
                elif action == "home":
                    if not DEBUG_MODE and 'ser' in globals():
                        home_angles = [x_mid, y_mid, 130, 0]
                        send_servo_command(home_angles, ser)
                        print("Robot moved to home position")
                        
        elif msg.topic == MQTT_TOPIC_CAMERA_SETTINGS:
            if "setting" in payload:
                setting = payload["setting"]
                if setting == "resolution":
                    if "width" in payload and "height" in payload:
                        global CAMERA_WIDTH, CAMERA_HEIGHT
                        CAMERA_WIDTH = payload["width"]
                        CAMERA_HEIGHT = payload["height"]
                        if 'cap' in globals():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                            print(f"Camera resolution changed to {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                elif setting == "framerate":
                    if "fps" in payload:
                        global frame_rate
                        frame_rate = payload["fps"]
                        print(f"Camera framerate changed to {frame_rate} FPS")
                        
        elif msg.topic == MQTT_TOPIC_SYSTEM_SETTINGS:
            if "setting" in payload:
                setting = payload["setting"]
                if setting == "similarity_threshold":
                    if "value" in payload:
                        global SIMILARITY_THRESHOLD
                        SIMILARITY_THRESHOLD = payload["value"]
                        print(f"Similarity threshold changed to {SIMILARITY_THRESHOLD}")
                elif setting == "smoothing_factor":
                    if "value" in payload:
                        global SMOOTHING_FACTOR
                        SMOOTHING_FACTOR = payload["value"]
                        print(f"Smoothing factor changed to {SMOOTHING_FACTOR}")
                        
    except Exception as e:
        print(f"Error processing MQTT message: {e}")
        print(f"Message payload: {msg.payload.decode()}")

def send_status_update():
    """Send current status via MQTT"""
    if mqtt_client and mqtt_client.is_connected():
        status_data = {
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
        print("Status update sent")

def setup_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}")
        
        # Wait for connection to be established
        connection_timeout = 10  # seconds
        start_time = time.time()
        while not mqtt_client.is_connected():
            if time.time() - start_time > connection_timeout:
                print("✗ MQTT connection timeout")
                return False
            time.sleep(0.1)
        
        # Send initial connection status with robot status
        mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps({
            "type": "robot_status",
            "status": "online",
            "title": "FaceBot Online",
            "body": "Your FaceBot system is now online and monitoring",
            "timestamp": datetime.now().isoformat()
        }))
        
        return True
    except Exception as e:
        print(f"✗ Failed to connect to MQTT broker: {e}")
        return False

def send_mqtt_notification(data):
    """Send a notification via MQTT"""
    if mqtt_client and mqtt_client.is_connected():
        try:
            payload = json.dumps(data)
            mqtt_client.publish(MQTT_TOPIC_NOTIFICATIONS, payload)
            print(f"✓ MQTT notification sent: {data['title']}")
            return True
        except Exception as e:
            print(f"✗ Failed to send MQTT notification: {e}")
    return False

def setup_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    mtcnn = MTCNN(keep_all=True, device=device)
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
    logging.info("✅Loaded faces.")
    return conn, cursor, known_embeddings, known_names

def send_email_async(cropped_face, subject="Unknown Person Detected"):
    def _send():
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = subject
            msg.attach(MIMEText("An unknown person was detected by your security system.", 'plain'))
            
            # Convert cropped face to RGB for PIL
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_face_rgb)
            
            # Create unique temp filename
            temp_path = os.path.join(SAVED_UNKNOWN_DIR, f"temp_cropped_face_{int(time.time())}.jpg")
            pil_image.save(temp_path)
            
            with open(temp_path, 'rb') as f:
                img_data = f.read()
            msg.attach(MIMEImage(img_data, name="unknown_face.jpg"))
            
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
    
    # Start email sending in background thread
    threading.Thread(target=_send, daemon=True).start()

def save_unknown_person(cropped_face, embedding, db_cursor):
    try:
        # Convert cropped face to JPEG
        _, img_encoded = cv2.imencode('.jpg', cropped_face)
        img_bytes = img_encoded.tobytes()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        # Insert into database
        db_cursor.execute("""
            INSERT INTO unknown_persons (image, embedding)
            VALUES (?, ?)
        """, (img_bytes, embedding_bytes))
        print("✅ Saved unknown person to database")
        return True
    except Exception as e:
        print(f"Error saving unknown person to database: {e}")
        return False

def map_face_to_servo(face_center, face_size, frame_width, frame_height):
    x, y = face_center
    face_width, face_height = face_size
    
    # Normalize coordinates (0-1)
    x_norm = x / frame_width
    y_norm = y / frame_height
    
    # Calculate servo angles (0-180)
    # X-axis: Inverted (face moves right → robot turns left)
    servo_x = int((1 - x_norm) * 180)
    
    # Y-axis: Not inverted (face moves up → robot looks up)
    servo_y = int( y_norm * 180)
    
    # Z-axis: Based on face size (normalized between 0.1 and 1.0)
    face_area = face_width * face_height
    frame_area = frame_width * frame_height
    size_ratio = face_area / frame_area
    
    # Normalize size ratio (adjust these values based on your observations)
    min_expected_size = 0.02  # Face size when very far
    max_expected_size = 0.3   # Face size when very close
    size_norm = (size_ratio - min_expected_size) / (max_expected_size - min_expected_size)
    size_norm = max(0.0, min(1.0, size_norm))  # Clamp between 0.1 and 1.0
    
    # Map to Z servo position (inverted: smaller face → lower Z value)
    servo_z = int(z_min + ((z_max-z_min) * 1.5) * size_norm )
    
    # Constrain to servo limits
    servo_x = max(x_min, min(x_max, servo_x))
    servo_y = max(y_min, min(y_max, servo_y))
    servo_z = max(z_min, min(z_max, servo_z))
    
    return [servo_x, servo_y, servo_z, 0]  # [x, y, z, claw]

def send_servo_command(angles, ser):
    if DEBUG_MODE:
        print(f"DEBUG: Servo angles: {angles}")
    else:
        try:
            ser.write(bytearray(angles))
            # print(f"Sent servo angles: {angles}")
        except Exception as e:
            print(f"Error sending servo command: {e}")

def send_unknown_face_notification(face_image, timestamp=None):
    """Send unknown face notification with image via MQTT"""
    if not mqtt_client or not mqtt_client.is_connected():
        print("MQTT client not connected, can't send unknown face notification")
        return False
    
    try:
        # Convert OpenCV BGR to RGB for better image quality
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to reduce payload size while maintaining aspect ratio
        max_dim = 400
        h, w = rgb_image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            rgb_image = cv2.resize(rgb_image, new_size)
        
        # Convert to JPEG format with balanced quality
        success, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print("Failed to encode image")
            return False
            
        # Convert to base64 string - Fix: buffer is already bytes, no need for tobytes()
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Create payload with image data
        payload = {
            "type": "unknown_face",
            "title": "Unknown Person Detected",
            "body": "An unknown person was detected by your FaceBot",
            "timestamp": timestamp or datetime.now().isoformat(),
            "image_data": {
                "format": "jpg",
                "encoding": "base64",
                "data": base64_image
            }
        }
        
        # Publish to MQTT with QoS 1 for better delivery guarantee
        result = mqtt_client.publish(
            MQTT_TOPIC_UNKNOWN_FACE, 
            json.dumps(payload),
            qos=1,
            retain=False
        )
        
        # Check if message was published
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print("✓ Unknown face notification sent via MQTT")
            # Also send a lightweight notification without image
            send_mqtt_notification({
                "type": "alert",
                "title": "Unknown Person Detected",
                "body": "Check the details for the captured image",
                "timestamp": datetime.now().isoformat()
            })
            return True
        else:
            print(f"✗ Failed to publish MQTT message, result code: {result.rc}")
            return False
    except Exception as e:
        print(f"✗ Failed to send unknown face notification: {e}")
        print(f"Error details: {str(e)}")
        return False

def handle_faces_request():
    try:
        print("Handling faces list request...")
        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person.id, person.name, faces.image 
            FROM person 
            JOIN faces ON person.id = faces.person_id
        """)
        faces = []
        for person_id, name, image_blob in cursor.fetchall():
            try:
                print(f"Processing face: {name} (ID: {person_id})")
                # Convert image blob to base64
                base64_img = base64.b64encode(image_blob).decode('utf-8')
                faces.append({
                    'id': person_id,
                    'name': name,
                    'image': base64_img
                })
                print(f"Successfully processed face: {name}")
            except Exception as e:
                print(f"Error processing face {person_id}: {e}")
                continue
        
        # Send faces list via MQTT
        if mqtt_client and mqtt_client.is_connected():
            payload = {
                'faces': faces,
                'timestamp': datetime.now().isoformat(),
                'count': len(faces)
            }
            json_payload = json.dumps(payload)
            print(f"Sending {len(faces)} faces to MQTT")
            print(f"Payload size: {len(json_payload)} bytes")
            
            result = mqtt_client.publish('facebot/faces/list', json_payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("✓ Successfully published faces list")
            else:
                print(f"✗ Failed to publish faces list, result code: {result.rc}")
        else:
            print("✗ MQTT client not connected, can't send faces list")
            
    except Exception as e:
        print(f"Error in handle_faces_request: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def handle_delete_face(face_id):
    try:
        print(f"Handling delete face request for ID: {face_id}")
        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()
        
        # Delete the face
        cursor.execute("DELETE FROM faces WHERE person_id = ?", (face_id,))
        cursor.execute("DELETE FROM person WHERE id = ?", (face_id,))
        conn.commit()
        
        print(f"✓ Successfully deleted face with ID: {face_id}")
        
        # Send updated faces list
        handle_faces_request()
        
    except Exception as e:
        print(f"Error deleting face: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    global servo_angles, prev_servo_angles
    
    # Setup MQTT
    mqtt_success = setup_mqtt()
    if not mqtt_success:
        print("WARNING: Continuing without MQTT functionality")
    
    # Setup
    device, mtcnn, resnet, preprocess = setup_models()
    conn, cursor, known_embeddings, known_names = load_known_faces(DB_PATH, device, resnet, preprocess)
    
    # Create directory for unknown faces if not exists
    if not os.path.exists(SAVED_UNKNOWN_DIR):
        os.makedirs(SAVED_UNKNOWN_DIR)

    # Initialize serial connection
    if not DEBUG_MODE:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)
            time.sleep(2)
            print(f"✅ Connected to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connected",
                    "body": "Successfully connected to robot hardware",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"❌ Failed to connect to serial port: {e}")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connection Failed",
                    "body": f"Failed to connect to robot hardware: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
            return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        if mqtt_success:
            send_mqtt_notification({
                "type": "camera_status",
                "title": "Camera Error",
                "body": "Could not access camera. Please check connections.",
                "timestamp": datetime.now().isoformat()
            })
        exit()

    # Reduce resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if mqtt_success:
        send_mqtt_notification({
            "type": "camera_status",
            "title": "Camera Connected",
            "body": "Successfully initialized camera",
            "timestamp": datetime.now().isoformat()
        })

    # Track processed faces to avoid duplicate captures
    processed_faces = set()

    # FPS control
    prev_time = 0
    frame_rate = 15  # Target FPS

    try:
        while True:
            current_time = time.time()
            if current_time - prev_time < 1.0 / frame_rate:
                continue
            prev_time = current_time

            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]

            # Resize for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                faces, probs = mtcnn.detect(rgb_small_frame)

            # Reset servo angles if no face detected
            if faces is None:
                servo_angles = [x_mid, y_mid, 130, 0]
                if servo_angles != prev_servo_angles:
                    if not DEBUG_MODE:
                        send_servo_command(servo_angles, ser)
                    prev_servo_angles = servo_angles.copy()
                continue

            if faces is not None:
                for i, box in enumerate(faces):
                    if probs[i] < 0.90:
                        continue
                    x1, y1, x2, y2 = [int(coord * 2) for coord in box]  # Scale back
                    face_img = frame[y1:y2, x1:x2]
                    face_width = x2 - x1
                    face_height = y2 - y1

                    try:
                        face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            face_embedding = resnet(face_tensor).cpu().numpy()

                        # Generate a unique hash for this face
                        face_hash = hash(tuple(face_embedding.tobytes()))

                        name = "Unknown"
                        for emb, known_name in zip(known_embeddings, known_names):
                            similarity = cosine_similarity(face_embedding, emb)
                            distance = 1 - similarity
                            if distance < SIMILARITY_THRESHOLD:
                                name = known_name
                                break

                        # Handle unknown face
                        if name == "Unknown" and face_hash not in processed_faces:
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            filename = f"Unknown_{timestamp}.jpg"
                            filepath = os.path.join(SAVED_UNKNOWN_DIR, filename)
                            
                            cropped_face = frame[y1:y2, x1:x2]
                            cv2.imwrite(filepath, cropped_face)
                            
                            if save_unknown_person(cropped_face, face_embedding, cursor):
                                conn.commit()
                                send_email_async(cropped_face)
                                
                                # Send MQTT notification with photo
                                if mqtt_success:
                                    send_unknown_face_notification(cropped_face)
                            
                            processed_faces.add(face_hash)

                        # Draw bounding box
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, 
                            name, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            color, 
                            2
                        )

                    except Exception as e:
                        print(f"Face processing error: {e}")

                    # Face following logic
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    new_angles = map_face_to_servo(
                        face_center, 
                        (face_width, face_height), 
                        frame_width, 
                        frame_height
                    )
                    
                    # Apply smoothing to Z-axis only
                    smoothed_z = int(prev_servo_angles[2] * (1-SMOOTHING_FACTOR) + new_angles[2] * SMOOTHING_FACTOR)
                    servo_angles = [new_angles[0], new_angles[1], smoothed_z, 0]

                    if servo_angles != prev_servo_angles:
                        if not DEBUG_MODE:
                            send_servo_command(servo_angles, ser)
                        prev_servo_angles = servo_angles.copy()

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add MQTT status updates periodically
            if mqtt_success and time.time() - prev_time > 300:  # Every 5 minutes
                send_status_update()
                
    except KeyboardInterrupt:
        print("Shutdown requested... exiting")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Send offline status before cleanup
        if mqtt_success and mqtt_client and mqtt_client.is_connected():
            try:
                # Send robot offline status
                mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps({
                    "type": "robot_status",
                    "status": "offline",
                    "title": "FaceBot Offline",
                    "body": "The FaceBot system is shutting down",
                    "timestamp": datetime.now().isoformat()
                }))
                # Wait a bit to ensure message is sent
                time.sleep(0.5)
            except Exception as e:
                print(f"Error sending offline status: {e}")
        
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'ser' in locals() and not DEBUG_MODE:
            ser.close()
        
        # Stop MQTT client
        if mqtt_success and mqtt_client:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                print("MQTT client disconnected")
            except Exception as e:
                print(f"Error disconnecting MQTT client: {e}")

if __name__ == "__main__":
    main()