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
import json
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import logging
import threading
import serial
import base64
import paho.mqtt.client as mqtt
from picamera2 import Picamera2, Preview
import traceback

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

# Raspberry Pi Performance Optimization
DETECTION_INTERVAL = 0.2  # Seconds between face detection attempts
FACE_DETECTION_SIZE = (320, 240)  # Lower resolution for face detection
DISPLAY_SIZE = (640, 480)  # Display resolution

# Robot Control Config
SERIAL_PORT = '/dev/ttyUSB0'  # Standard Raspberry Pi USB port
SERIAL_BAUDRATE = 115200
DEBUG_MODE = False

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"  # Public broker for testing
MQTT_PORT = 1883
MQTT_TOPIC_NOTIFICATIONS = "facebot/notifications"
MQTT_TOPIC_UNKNOWN_FACE = "facebot/unknown_face"
MQTT_CLIENT_ID = "facebot_raspberry_pi"

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("facebot.log")
    ]
)
logger = logging.getLogger("FaceBot")

def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to topics if needed
    client.subscribe(f"{MQTT_TOPIC_NOTIFICATIONS}/command")

def on_message(client, userdata, msg):
    logger.info(f"Received message on {msg.topic}: {msg.payload.decode()}")
    # Handle commands from mobile app if needed
    try:
        payload = json.loads(msg.payload.decode())
        # Process commands here
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def setup_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        logger.info(f"✅ Connected to MQTT broker at {MQTT_BROKER}")
        
        # Send initial connection status
        send_mqtt_notification({
            "type": "status",
            "title": "FaceBot Online",
            "body": "Your FaceBot system is now online and monitoring",
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to MQTT broker: {e}")
        return False

def send_mqtt_notification(data):
    """Send a notification via MQTT"""
    if mqtt_client and mqtt_client.is_connected():
        try:
            payload = json.dumps(data)
            mqtt_client.publish(MQTT_TOPIC_NOTIFICATIONS, payload)
            logger.info(f"✅ MQTT notification sent: {data['title']}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send MQTT notification: {e}")
    return False

def send_unknown_face_notification(face_image, timestamp=None):
    """Send unknown face notification with image via MQTT"""
    if not mqtt_client or not mqtt_client.is_connected():
        logger.warning("MQTT client not connected, can't send unknown face notification")
        return False
    
    try:
        # Convert OpenCV BGR to RGB for better image quality
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to reduce payload size (300px max dimension)
        max_dim = 300
        h, w = rgb_image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            rgb_image = cv2.resize(rgb_image, new_size)
        
        # Convert to JPEG format with lower quality to save bandwidth
        _, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        
        # Convert to base64 string
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Log size for debugging
        logger.debug(f"Image base64 size: {len(base64_image)} bytes")
        
        # Create payload
        payload = {
            "type": "unknown_face",
            "title": "Unknown Person Detected",
            "body": "An unknown person was detected by your FaceBot",
            "timestamp": timestamp or datetime.now().isoformat(),
            "image": base64_image  # This is the key the Flutter app is looking for
        }
        
        # Publish to MQTT - use the specific unknown face topic
        result = mqtt_client.publish(MQTT_TOPIC_UNKNOWN_FACE, json.dumps(payload))
        
        # Check if message was published
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("✅ Unknown face notification sent via MQTT")
            # Also send a lightweight notification without image
            send_mqtt_notification({
                "type": "alert",
                "title": "Unknown Person Detected",
                "body": "Check the details for the captured image",
                "timestamp": datetime.now().isoformat()
            })
            return True
        else:
            logger.error(f"❌ Failed to publish MQTT message, result code: {result.rc}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to send unknown face notification: {e}")
        traceback.print_exc()
        return False

def setup_models():
    """Set up ML models with performance optimizations for Raspberry Pi"""
    # Force CPU for better compatibility on Raspberry Pi
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize MTCNN with lower threshold for better performance
    # Only keep best face to reduce computation
    mtcnn = MTCNN(
        keep_all=False,  # Only detect primary face for performance
        min_face_size=40,  # Set minimum face size to detect
        thresholds=[0.6, 0.7, 0.7],  # Slightly lower thresholds
        factor=0.709,  # Increase for faster detection
        post_process=True,
        device=device
    )
    
    # Load the face recognition model
    # Use margin to include more of the face for better recognition
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return device, mtcnn, resnet, preprocess

def load_known_faces(db_path, device, resnet, preprocess):
    """Load known faces from database with error handling"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        known_embeddings = []
        known_names = []

        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                image BLOB NOT NULL,
                FOREIGN KEY (person_id) REFERENCES person(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unknown_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB NOT NULL,
                embedding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

        cursor.execute("""
            SELECT person.id, person.name, faces.image
            FROM person
            JOIN faces ON person.id = faces.person_id
        """)
        rows = cursor.fetchall()

        logger.info(f"Loading {len(rows)} known faces from database...")
        for person_id, person_name, image_blob in rows:
            try:
                # Load image from blob
                image = Image.open(io.BytesIO(image_blob)).convert("RGB")
                
                # Process the face image
                face_tensor = preprocess(np.array(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()
                
                # Store the embedding and name
                known_embeddings.append(embedding)
                known_names.append(person_name)
                logger.debug(f"Loaded face for {person_name}")
            except Exception as e:
                logger.error(f"Error loading image for {person_name}: {e}")

        logger.info(f"✅ Loaded {len(known_embeddings)} faces successfully")
        return conn, cursor, known_embeddings, known_names
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Create an empty database if it doesn't exist
        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            return conn, cursor, [], []
        raise

def send_email_async(cropped_face, subject="Unknown Person Detected"):
    """Send email notification in background thread"""
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
            logger.info(f"📧 Email sent with cropped face.")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Start email sending in background thread
    threading.Thread(target=_send, daemon=True).start()

def save_unknown_person(cropped_face, embedding, db_cursor):
    """Save unknown person to database"""
    try:
        # Convert cropped face to JPEG with lower quality for storage efficiency
        _, img_encoded = cv2.imencode('.jpg', cropped_face, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_bytes = img_encoded.tobytes()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        # Insert into database
        db_cursor.execute("""
            INSERT INTO unknown_persons (image, embedding)
            VALUES (?, ?)
        """, (img_bytes, embedding_bytes))
        logger.info("✅ Saved unknown person to database")
        return True
    except Exception as e:
        logger.error(f"Error saving unknown person to database: {e}")
        return False

def map_face_to_servo(face_center, face_size, frame_width, frame_height):
    """Map face position to servo angles"""
    x, y = face_center
    face_width, face_height = face_size
    
    # Normalize coordinates (0-1)
    x_norm = x / frame_width
    y_norm = y / frame_height
    
    # Calculate servo angles (0-180)
    # X-axis: Inverted (face moves right → robot turns left)
    servo_x = int((1 - x_norm) * (x_max - x_min) + x_min)
    
    # Y-axis: Not inverted (face moves up → robot looks up)
    servo_y = int(y_norm * (y_max - y_min) + y_min)
    
    # Z-axis: Based on face size (normalized between 0.1 and 1.0)
    face_area = face_width * face_height
    frame_area = frame_width * frame_height
    size_ratio = face_area / frame_area
    
    # Normalize size ratio (adjust these values based on your observations)
    min_expected_size = 0.02  # Face size when very far
    max_expected_size = 0.3   # Face size when very close
    size_norm = (size_ratio - min_expected_size) / (max_expected_size - min_expected_size)
    size_norm = max(0.0, min(1.0, size_norm))  # Clamp between 0 and 1.0
    
    # Map to Z servo position (inverted: smaller face → lower Z value)
    servo_z = int(z_min + (z_max - z_min) * size_norm * 1.5)
    
    # Constrain to servo limits
    servo_x = max(x_min, min(x_max, servo_x))
    servo_y = max(y_min, min(y_max, servo_y))
    servo_z = max(z_min, min(z_max, servo_z))
    
    return [servo_x, servo_y, servo_z, 0]  # [x, y, z, claw]

def send_servo_command(angles, ser):
    """Send angles to Arduino/servo controller"""
    if DEBUG_MODE:
        logger.debug(f"DEBUG: Servo angles: {angles}")
    else:
        try:
            ser.write(bytearray(angles))
            logger.debug(f"Sent servo angles: {angles}")
        except Exception as e:
            logger.error(f"Error sending servo command: {e}")

def setup_picamera():
    """Initialize PiCamera2 with optimal settings for face detection"""
    try:
        # Initialize camera
        picam2 = Picamera2()
        
        # Configure video capture
        # Use smaller resolution for better performance
        # Define configurations
        preview_config = picam2.create_preview_configuration(
            main={"size": DISPLAY_SIZE, "format": "RGB888"},
            buffer_count=2  # Reduce buffer for lower latency
        )
        
        # Apply configuration
        picam2.configure(preview_config)
        
        # Start camera
        picam2.start()
        
        # Allow camera to warm up
        time.sleep(1)
        
        logger.info("✅ PiCamera2 initialized successfully")
        return picam2
    except Exception as e:
        logger.error(f"❌ Failed to initialize PiCamera2: {e}")
        raise

def main():
    global servo_angles, prev_servo_angles
    
    # Setup MQTT
    mqtt_success = setup_mqtt()
    if not mqtt_success:
        logger.warning("Continuing without MQTT functionality")
    
    # Setup ML models
    device, mtcnn, resnet, preprocess = setup_models()
    conn, cursor, known_embeddings, known_names = load_known_faces(DB_PATH, device, resnet, preprocess)
    
    # Create directory for unknown faces if not exists
    if not os.path.exists(SAVED_UNKNOWN_DIR):
        os.makedirs(SAVED_UNKNOWN_DIR)

    # Initialize serial connection
    global DEBUG_MODE  # Declare global at the start of the function
    
    if not DEBUG_MODE:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)
            time.sleep(2)
            logger.info(f"✅ Connected to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connected",
                    "body": "Successfully connected to robot hardware",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to serial port: {e}")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connection Failed",
                    "body": f"Failed to connect to robot hardware: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
            if not DEBUG_MODE:
                # If not in debug mode, enable it to continue without hardware
                logger.warning("Switching to DEBUG_MODE due to hardware connection failure")
                DEBUG_MODE = True

    # Initialize camera
    try:
        picam2 = setup_picamera()
        
        if mqtt_success:
            send_mqtt_notification({
                "type": "camera_status",
                "title": "Camera Ready",
                "body": "Camera initialized successfully",
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        
        if mqtt_success:
            send_mqtt_notification({
                "type": "camera_status",
                "title": "Camera Error",
                "body": "Could not initialize camera. Please check connections.",
                "timestamp": datetime.now().isoformat()
            })
        exit(1)

    # Track processed faces to avoid duplicate captures
    processed_faces = set()

    # Set up window for display if needed
    if not DEBUG_MODE:
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", 640, 480)

    # Detection timing and FPS control
    last_detection_time = 0
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Notification cooldown for unknown faces (seconds)
    notification_cooldown = 30
    last_notification_time = 0

    # Initialize "no face detected" counter to reduce servo jitter
    no_face_counter = 0
    face_detected = False

    logger.info("Starting main detection loop")
    
    try:
        while True:
            try:
                # Get current time to manage detection intervals
                current_time = time.time()
                
                # Calculate FPS every second
                frame_count += 1
                if current_time - fps_start_time >= 1:
                    fps = frame_count / (current_time - fps_start_time)
                    frame_count = 0
                    fps_start_time = current_time
                
                # Capture frame from PiCamera
                frame = picam2.capture_array()
                
                # Convert to BGR for OpenCV compatibility if needed
                if frame.shape[2] == 3:  # Already RGB
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                frame_height, frame_width = frame_bgr.shape[:2]
                
                # Create a copy for display
                display_frame = frame_bgr.copy()
                
                # Only perform face detection at defined intervals
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    last_detection_time = current_time
                    
                    # Convert to RGB for MTCNN
                    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Resize for faster detection
                    small_frame = cv2.resize(rgb_frame, FACE_DETECTION_SIZE)
                    
                    # Detect faces
                    # Wrap in try block to handle detection errors
                    try:
                        boxes, probs = mtcnn.detect(small_frame)
                        face_detected = boxes is not None and len(boxes) > 0
                    except Exception as e:
                        logger.error(f"Face detection error: {e}")
                        boxes = None
                        face_detected = False
                    
                    # If no face detected, increment counter
                    if not face_detected:
                        no_face_counter += 1
                        
                        # Only reset servo position after several consecutive frames without faces
                        if no_face_counter >= 5:  # About 1 second without faces
                            servo_angles = [x_mid, y_mid, 130, 0]
                            if servo_angles != prev_servo_angles:
                                if not DEBUG_MODE:
                                    send_servo_command(servo_angles, ser)
                                prev_servo_angles = servo_angles.copy()
                    else:
                        # Reset counter when face detected
                        no_face_counter = 0
                    
                    # Process detected faces
                    if boxes is not None and len(boxes) > 0:
                        # Scale coordinates back to original size
                        scale_x = frame_width / FACE_DETECTION_SIZE[0]
                        scale_y = frame_height / FACE_DETECTION_SIZE[1]
                        
                        # Get the face with highest probability
                        best_idx = np.argmax(probs)
                        box = boxes[best_idx]
                        
                        # Scale box coordinates to match original frame
                        x1 = int(max(0, box[0] * scale_x))
                        y1 = int(max(0, box[1] * scale_y))
                        x2 = int(min(frame_width, box[2] * scale_x))
                        y2 = int(min(frame_height, box[3] * scale_y))
                        
                        # Check if box dimensions are valid
                        if x2 > x1 and y2 > y1:
                            face_img = frame_bgr[y1:y2, x1:x2]
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            # Only process face if it's a reasonable size
                            if face_width > 30 and face_height > 30:
                                try:
                                    # Convert to PIL Image for the model
                                    pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                    
                                    # Get face embedding
                                    face_tensor = preprocess(pil_face).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        face_embedding = resnet(face_tensor).cpu().numpy()
                                    
                                    # Generate a unique hash for this face
                                    face_hash = str(hash(face_embedding.tobytes()))[:10]
                                    
                                    # Initialize name as unknown
                                    name = "Unknown"
                                    best_match_score = float('inf')
                                    
                                    # Compare with known faces
                                    for emb, known_name in zip(known_embeddings, known_names):
                                        similarity = cosine_similarity(face_embedding, emb)
                                        distance = 1 - similarity[0][0]
                                        
                                        if distance < SIMILARITY_THRESHOLD and distance < best_match_score:
                                            name = known_name
                                            best_match_score = distance
                                    
                                    # Handle unknown face
                                    if name == "Unknown" and face_hash not in processed_faces:
                                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                                        filename = f"Unknown_{timestamp}.jpg"
                                        filepath = os.path.join(SAVED_UNKNOWN_DIR, filename)
                                        
                                        # Save face image
                                        cv2.imwrite(filepath, face_img)
                                        logger.info(f"Unknown face saved to {filepath}")
                                        
                                        # Save to DB and send notifications
                                        if save_unknown_person(face_img, face_embedding, cursor):
                                            conn.commit()
                                            
                                            # Send email in background
                                            send_email_async(face_img)
                                            
                                            # Send MQTT notification with cooldown
                                            current_time = time.time()
                                            if current_time - last_notification_time > notification_cooldown:
                                                if mqtt_success:
                                                    success = send_unknown_face_notification(face_img)
                                                    logger.info(f"MQTT unknown face notification sent: {success}")
                                                last_notification_time = current_time
                                        
                                        # Mark as processed to avoid duplicate alerts
                                        processed_faces.add(face_hash)
                                        
                                        # Limit size of processed_faces set
                                        if len(processed_faces) > 100:
                                            # Remove oldest entries
                                            processed_faces = set(list(processed_faces)[-50:])
                                    
                                    # Draw bounding box and name
                                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(
                                        display_frame, 
                                        name, 
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, 
                                        color, 
                                        2
                                    )
                                    
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
                                    
                                    # Only update servos if angle change is significant (reduces jitter)
                                    angle_diff = sum(abs(a - b) for a, b in zip(servo_angles, prev_servo_angles))
                                    if angle_diff > 3:  # Only move if total angle change > 3 degrees
                                        if not DEBUG_MODE:
                                            send_servo_command(servo_angles, ser)
                                        prev_servo_angles = servo_angles.copy()
                                
                                except Exception as e:
                                    logger.error(f"Face processing error: {e}")
                                    traceback.print_exc()
                
                # Add FPS counter to frame
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Display frame
                cv2.imshow("Face Recognition", display_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Add small delay to reduce CPU usage
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                # Continue running to recover from errors
                time.sleep(1)
    
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if 'picam2' in locals():
            picam2.close()
        
        cv2.destroyAllWindows()
        conn.close()
        
        if not DEBUG_MODE and 'ser' in locals():
            # Center servos before exit
            send_servo_command([x_mid, y_mid, 130, 0], ser)
            time.sleep(0.5)
            ser.close()
        
        # Stop MQTT client
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.info("MQTT client disconnected")
        
        logger.info("FaceBot shutdown complete")

if __name__ == "__main__":
    # Check if running on Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                logger.info(f"Running on {model.strip()}")
            else:
                logger.warning(f"Not running on a Raspberry Pi: {model.strip()}")
    except Exception as e:
        logger.warning(f"Could not determine if running on Raspberry Pi: {e}")
    
    # Start the main program
    main()