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

# Configuration
DB_PATH = "faces.db"
SAVED_UNKNOWN_DIR = "Unknown"
EMAIL_ADDRESS = "smartfacebot@gmail.com"
EMAIL_PASSWORD = "pbmd izfw indc pmbm" 
RECEIVER_EMAIL = "prithakhamtu@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SIMILARITY_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.3  # For smoother Z-axis movements
EMAIL_COOLDOWN = 5  # Cooldown time in seconds for email notifications

# Robot Control Config
SERIAL_PORT = '/dev/cu.usbserial-1120'
SERIAL_BAUDRATE = 115200
DEBUG_MODE = False

# Multi-face detection settings
MAX_FACES_FOR_TRACKING = 1  # Only track when this many faces or fewer are detected
MULTI_FACE_COOLDOWN = 2.0   # Seconds to wait before resuming tracking after multiple faces detected

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
MQTT_TOPIC_SIGNUPS = "facebot/signups"

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

# Multi-face tracking variables
last_multi_face_time = 0
is_tracking_paused = False

# MQTT Client
mqtt_client = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceBot")

def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to command topic for remote control
    client.subscribe(f"{MQTT_TOPIC_COMMAND}")
    client.subscribe(MQTT_TOPIC_SIGNUPS)
    # Publish online status
    client.publish(MQTT_TOPIC_STATUS, json.dumps({
        "type": "robot_status",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }))

# Add this function to handle signup emails
def handle_signup_email(payload):
    try:
        data = json.loads(payload)
        if data.get('type') == 'new_signup':
            email = data['email']
            timestamp = data.get('timestamp', datetime.now().isoformat())
            action = data.get('action', 'new_signup')
            
            if action == 'account_created':
                # Send welcome email for new account
                subject = "Welcome to FaceBot - Account Created Successfully!"
                body = f"""
                Dear User,

                Welcome to FaceBot! Your account has been successfully created.

                Thank you for choosing FaceBot for your security needs. You can now:
                - Monitor your security system
                - Receive real-time notifications
                - Manage your face recognition settings
                - Access your security dashboard

                If you didn't create this account, please contact support immediately.

                Account Creation Time: {timestamp}

                Best regards,
                The FaceBot Team
                """
            else:
                # Regular signup notification
                subject = "Welcome to FaceBot!"
                body = f"""
                Thank you for signing up with FaceBot!
                
                Your account has been successfully created.
                
                If you didn't request this, please contact support immediately.
                
                Signup time: {timestamp}
                """
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Sent welcome email to {email}")
            print(f"✓ Sent welcome email to {email}")
            
            # Also send notification to admin
            admin_msg = MIMEMultipart()
            admin_msg['From'] = EMAIL_ADDRESS
            admin_msg['To'] = RECEIVER_EMAIL
            admin_msg['Subject'] = f"New FaceBot Signup: {email}"
            admin_msg.attach(MIMEText(f"""
            A new user has signed up for FaceBot:
            
            Email: {email}
            Timestamp: {timestamp}
            Action: {action}
            
            This is an automated notification from the FaceBot system.
            """, 'plain'))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(admin_msg)
            
            logger.info(f"Sent admin notification about new signup: {email}")
            print(f"✓ Sent admin notification about new signup: {email}")
            
    except Exception as e:
        logger.error(f"Error processing signup email: {e}")
        print(f"✗ Error processing signup email: {e}")

def on_message(client, userdata, msg):
    logger.info(f"Received message on {msg.topic}: {msg.payload.decode()}")
    try:
        payload = json.loads(msg.payload.decode())

        if msg.topic == MQTT_TOPIC_SIGNUPS:
            handle_signup_email(payload)
            
        return
    
        if msg.topic == MQTT_TOPIC_COMMAND:
            if "command" in payload:
                cmd = payload["command"]
                if cmd == "restart":
                    logger.info("Received restart command")
                    # Implement restart logic here or send signal to systemd
                elif cmd == "shutdown":
                    logger.info("Received shutdown command")
                    # Implement shutdown logic here
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
                                logger.info(f"Robot moved to angles: {angles}")
                elif action == "home":
                    if not DEBUG_MODE and 'ser' in globals():
                        home_angles = [x_mid, y_mid, 130, 0]
                        send_servo_command(home_angles, ser)
                        logger.info("Robot moved to home position")
                        
        elif msg.topic == MQTT_TOPIC_CAMERA_SETTINGS:
            if "setting" in payload:
                setting = payload["setting"]
                if setting == "resolution":
                    if "width" in payload and "height" in payload:
                        global CAMERA_WIDTH, CAMERA_HEIGHT
                        CAMERA_WIDTH = payload["width"]
                        CAMERA_HEIGHT = payload["height"]
                        logger.info(f"Camera resolution changed to {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                elif setting == "framerate":
                    if "fps" in payload:
                        global FRAME_RATE
                        FRAME_RATE = payload["fps"]
                        logger.info(f"Camera framerate changed to {FRAME_RATE} FPS")
                        
        elif msg.topic == MQTT_TOPIC_SYSTEM_SETTINGS:
            if "setting" in payload:
                setting = payload["setting"]
                if setting == "similarity_threshold":
                    if "value" in payload:
                        global SIMILARITY_THRESHOLD
                        SIMILARITY_THRESHOLD = payload["value"]
                        logger.info(f"Similarity threshold changed to {SIMILARITY_THRESHOLD}")
                elif setting == "smoothing_factor":
                    if "value" in payload:
                        global SMOOTHING_FACTOR
                        SMOOTHING_FACTOR = payload["value"]
                        logger.info(f"Smoothing factor changed to {SMOOTHING_FACTOR}")
                elif setting == "notification_cooldown":
                    if "value" in payload:
                        global notification_cooldown
                        notification_cooldown = payload["value"]
                        logger.info(f"Notification cooldown changed to {notification_cooldown} seconds")
                elif setting == "max_faces_for_tracking":
                    if "value" in payload:
                        global MAX_FACES_FOR_TRACKING
                        MAX_FACES_FOR_TRACKING = payload["value"]
                        logger.info(f"Max faces for tracking changed to {MAX_FACES_FOR_TRACKING}")
                elif setting == "multi_face_cooldown":
                    if "value" in payload:
                        global MULTI_FACE_COOLDOWN
                        MULTI_FACE_COOLDOWN = payload["value"]
                        logger.info(f"Multi-face cooldown changed to {MULTI_FACE_COOLDOWN} seconds")
                        
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def send_status_update():
    """Send current status via MQTT"""
    if mqtt_client and mqtt_client.is_connected():
        status_data = {
            "type": "robot_status",
            "status": "online",
            "uptime": time.time() - startup_time,
            "timestamp": datetime.now().isoformat(),
            "tracking_paused": is_tracking_paused
        }
        mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
        logger.info("Status update sent")

def setup_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        logger.info(f"✓ Connected to MQTT broker at {MQTT_BROKER}")
        
        # Subscribe to all command topics
        mqtt_client.subscribe(MQTT_TOPIC_COMMAND)
        mqtt_client.subscribe(MQTT_TOPIC_ROBOT_CONTROL)
        mqtt_client.subscribe(MQTT_TOPIC_CAMERA_SETTINGS)
        mqtt_client.subscribe(MQTT_TOPIC_SYSTEM_SETTINGS)
        
        # Send initial connection status
        send_mqtt_notification({
            "type": "status",
            "title": "FaceBot Online",
            "body": "Your FaceBot system is now online and monitoring",
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to MQTT broker: {e}")
        return False

def send_mqtt_notification(data):
    """Send a notification via MQTT"""
    if mqtt_client and mqtt_client.is_connected():
        try:
            payload = json.dumps(data)
            mqtt_client.publish(MQTT_TOPIC_NOTIFICATIONS, payload)
            logger.info(f"✓ MQTT notification sent: {data['title']}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to send MQTT notification: {e}")
    return False

def send_unknown_face_notification(face_image, timestamp=None):
    """Send unknown face notification with image via MQTT"""
    if not mqtt_client or not mqtt_client.is_connected():
        logger.warning("MQTT client not connected, can't send unknown face notification")
        return False
    
    try:
        # Convert OpenCV BGR to RGB for better image quality
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to reduce payload size
        max_dim = 400
        h, w = rgb_image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            rgb_image = cv2.resize(rgb_image, new_size)
        
        # Convert to JPEG format with balanced quality
        success, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            logger.error("Failed to encode image")
            return False
            
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
            "image": base64_image
        }
        
        # Publish to MQTT - use the specific unknown face topic
        result = mqtt_client.publish(MQTT_TOPIC_UNKNOWN_FACE, json.dumps(payload))
        
        # Check if message was published
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("✓ Unknown face notification sent via MQTT")
            # Also send a lightweight notification without image
            send_mqtt_notification({
                "type": "alert",
                "title": "Unknown Person Detected",
                "body": "Check the details for the captured image",
                "timestamp": datetime.now().isoformat()
            })
            return True
        else:
            logger.error(f"✗ Failed to publish MQTT message, result code: {result.rc}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to send unknown face notification: {e}")
        logger.exception("Full exception details:")
        return False

def setup_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
            logger.error(f"Error loading image for {person_name}: {e}")

    logger.info(f"✓ Loaded {len(known_names)} known faces.")
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
            logger.info(f"✓ Email sent with cropped face.")
        except Exception as e:
            logger.error(f"✗ Failed to send email: {e}")
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
        logger.info("✓ Saved unknown person to database")
        return True
    except Exception as e:
        logger.error(f"✗ Error saving unknown person to database: {e}")
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
    servo_y = int(y_norm * 180)
    
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
    servo_z = int(z_min + ((z_max-z_min) * 1.5) * size_norm)
    
    # Constrain to servo limits
    servo_x = max(x_min, min(x_max, servo_x))
    servo_y = max(y_min, min(y_max, servo_y))
    servo_z = max(z_min, min(z_max, servo_z))
    
    return [servo_x, servo_y, servo_z, 0]  # [x, y, z, claw]

def send_servo_command(angles, ser):
    if DEBUG_MODE:
        logger.debug(f"DEBUG: Servo angles: {angles}")
    else:
        try:
            ser.write(bytearray(angles))
            logger.debug(f"Sent servo angles: {angles}")
        except Exception as e:
            logger.error(f"✗ Error sending servo command: {e}")

def check_multiple_faces(faces, probs):
    """
    Check if multiple valid faces are detected and handle tracking pause logic
    Returns: (should_track, primary_face_index)
    """
    global last_multi_face_time, is_tracking_paused
    
    # Count valid faces (above confidence threshold)
    valid_faces = []
    for i, (face, prob) in enumerate(zip(faces, probs)):
        if prob >= 0.90:  # Same threshold as main detection
            valid_faces.append(i)
    
    current_time = time.time()
    
    # If more than allowed faces detected
    if len(valid_faces) > MAX_FACES_FOR_TRACKING:
        last_multi_face_time = current_time
        if not is_tracking_paused:
            is_tracking_paused = True
            logger.info(f"⚠️  Multiple faces detected ({len(valid_faces)}), pausing robot tracking for safety")
            
            # Send MQTT notification about multiple faces
            if mqtt_client and mqtt_client.is_connected():
                send_mqtt_notification({
                    "type": "tracking_status",
                    "title": "Multiple Faces Detected",
                    "body": f"Robot tracking paused - {len(valid_faces)} faces detected",
                    "timestamp": datetime.now().isoformat()
                })
        
        return False, None
    
    # Check if we should resume tracking after cooldown
    elif len(valid_faces) <= MAX_FACES_FOR_TRACKING and is_tracking_paused:
        time_since_multi_face = current_time - last_multi_face_time
        
        if time_since_multi_face >= MULTI_FACE_COOLDOWN:
            is_tracking_paused = False
            logger.info("✅ Resuming robot tracking - single face detected after cooldown")
            
            # Send MQTT notification about resuming tracking
            if mqtt_client and mqtt_client.is_connected():
                send_mqtt_notification({
                    "type": "tracking_status", 
                    "title": "Tracking Resumed",
                    "body": "Robot tracking resumed - safe to track single face",
                    "timestamp": datetime.now().isoformat()
                })
        else:
            # Still in cooldown period
            remaining_cooldown = MULTI_FACE_COOLDOWN - time_since_multi_face
            logger.debug(f"Still in cooldown period, {remaining_cooldown:.1f}s remaining")
            return False, None
    
    # Return tracking permission and index of first valid face (if any)
    if len(valid_faces) > 0 and not is_tracking_paused:
        return True, valid_faces[0]
    else:
        return False, None

def health_check_thread(stop_event):
    """Thread to periodically send health check status updates"""
    while not stop_event.is_set():
        try:
            # Send a status update every 5 minutes
            send_status_update()
        except Exception as e:
            logger.error(f"✗ Error in health check: {e}")
        
        # Sleep for 5 minutes
        for _ in range(300):  # 5 minutes in seconds
            if stop_event.is_set():
                break
            time.sleep(1)

def fetch_current_user_email(max_retries=3, retry_delay=2):
    """Fetch the current user's email from the database through MQTT with retries"""
    for attempt in range(max_retries):
        try:
            print(f"\n=== Fetching Current User Email (Attempt {attempt + 1}/{max_retries}) ===")
            # Create a temporary MQTT client for fetching user data
            temp_client = mqtt.Client(client_id=f"facebot_fetch_user_{int(time.time())}")
            
            # Create an event to wait for response
            email_received = threading.Event()
            current_email = [None]  # Use list to store email in callback
            
            def on_connect(client, userdata, flags, rc):
                print(f"✓ Connected to MQTT broker with result code {rc}")
                # Subscribe to the response topic first
                client.subscribe('facebot/current_user')
                print("✓ Subscribed to 'facebot/current_user' topic")
                
                # Wait a moment for subscription to be established
                time.sleep(0.5)
                
                # Then request current user email
                print("\nRequesting current user email...")
                client.publish('facebot/request_current_user', json.dumps({
                    'type': 'request_current_user',
                    'timestamp': datetime.now().isoformat()
                }))
                print("✓ Published request to 'facebot/request_current_user'")
            
            def on_message(client, userdata, msg):
                try:
                    print(f"\nReceived message on topic: {msg.topic}")
                    data = json.loads(msg.payload.decode())
                    print(f"Message data: {data}")
                    
                    if msg.topic == 'facebot/current_user' and data.get('type') == 'current_user_email':
                        current_email[0] = data.get('email')
                        print(f"✓ Received user email: {current_email[0]}")
                        email_received.set()
                except Exception as e:
                    print(f"✗ Error processing user email message: {e}")
            
            # Set up callbacks
            temp_client.on_connect = on_connect
            temp_client.on_message = on_message
            
            # Connect and start loop
            temp_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            temp_client.loop_start()
            
            # Wait for response with longer timeout
            print("\nWaiting for response...")
            if email_received.wait(timeout=10.0):  # Increased timeout to 10 seconds
                print(f"✓ Successfully received email: {current_email[0]}")
                temp_client.loop_stop()
                temp_client.disconnect()
                return current_email[0]
            else:
                print(f"✗ Timeout waiting for current user email (Attempt {attempt + 1})")
                temp_client.loop_stop()
                temp_client.disconnect()
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                
        except Exception as e:
            print(f"✗ Error fetching current user email (Attempt {attempt + 1}): {e}")
            if 'temp_client' in locals():
                try:
                    temp_client.loop_stop()
                    temp_client.disconnect()
                except:
                    pass
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print("=== Failed to fetch current user email after all attempts ===\n")
    return None

# Updated main function approach
def fetch_current_user_email_after_mqtt():
    """Fetch user email after MQTT client is established"""
    global mqtt_client
    
    if not mqtt_client or not mqtt_client.is_connected():
        print("✗ Main MQTT client not connected")
        return None
    
    print("\n=== Fetching Current User Email (After MQTT Setup) ===")
    
    # Create an event to wait for response
    email_received = threading.Event()
    current_email = [None]
    
    def check_for_user_email(client, userdata, msg):
        try:
            if msg.topic == 'facebot/current_user':
                data = json.loads(msg.payload.decode())
                print(f"Received user email data: {data}")
                
                if data.get('type') == 'current_user_email':
                    current_email[0] = data.get('email')
                    print(f"✓ Received user email: {current_email[0]}")
                    email_received.set()
        except Exception as e:
            print(f"✗ Error processing user email: {e}")
    
    # Store the original message handler
    original_handler = mqtt_client.on_message
    
    # Create a combined handler
    def combined_handler(client, userdata, msg):
        # First call the original handler
        if original_handler:
            original_handler(client, userdata, msg)
        # Then check for user email
        check_for_user_email(client, userdata, msg)
    
    # Set the combined handler
    mqtt_client.on_message = combined_handler
    
    # Subscribe and request
    mqtt_client.subscribe('facebot/current_user')
    print("✓ Subscribed to current user topic")
    
    time.sleep(0.5)  # Brief pause
    
    mqtt_client.publish('facebot/request_current_user', json.dumps({
        'type': 'request_current_user',
        'timestamp': datetime.now().isoformat()
    }))
    print("✓ Requested current user email")
    
    # Wait for response
    if email_received.wait(timeout=10.0):
        result = current_email[0]
        print(f"✓ Got email: {result}")
    else:
        result = None
        print("✗ Timeout waiting for email")
    
    # Restore original handler
    mqtt_client.on_message = original_handler
    
    return result

def main():
    global servo_angles, prev_servo_angles, startup_time, last_multi_face_time, is_tracking_paused
    
    # Record startup time for uptime tracking
    startup_time = time.time()
    
    print("\n=== FaceBot Startup ===")
    logger.info("Starting FaceBot in headless mode with multi-face safety")
    
    # Fetch current user's email at startup with retries
    print("\nFetching current user email...")
    current_user_email = fetch_current_user_email(max_retries=3, retry_delay=2)
    
    if current_user_email:
        print(f"\n✓ Current user email: {current_user_email}")
        logger.info(f"Current user email: {current_user_email}")
        # Update RECEIVER_EMAIL with current user's email
        global RECEIVER_EMAIL
        RECEIVER_EMAIL = current_user_email
        print(f"✓ Updated RECEIVER_EMAIL to: {RECEIVER_EMAIL}")
    else:
        print("\n✗ Could not fetch current user email, using default email")
        logger.warning("Could not fetch current user email, using default email")
        print(f"Default email: {RECEIVER_EMAIL}")
    
    # Ensure directories exist
    if not os.path.exists(SAVED_UNKNOWN_DIR):
        os.makedirs(SAVED_UNKNOWN_DIR)
        logger.info(f"Created directory: {SAVED_UNKNOWN_DIR}")
    
    # Setup MQTT
    mqtt_success = setup_mqtt()
    if not mqtt_success:
        logger.warning("WARNING: Continuing without MQTT functionality")
    
    # Setup ML models
    device, mtcnn, resnet, preprocess = setup_models()
    conn, cursor, known_embeddings, known_names = load_known_faces(DB_PATH, device, resnet, preprocess)
    
    # Initialize serial connection
    if not DEBUG_MODE:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)
            time.sleep(2)
            logger.info(f"✓ Connected to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connected",
                    "body": "Successfully connected to robot hardware",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"✗ Failed to connect to serial port: {e}")
            
            if mqtt_success:
                send_mqtt_notification({
                    "type": "hardware_status",
                    "title": "Robot Connection Failed",
                    "body": f"Failed to connect to robot hardware: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
            if not DEBUG_MODE:
                # Exit if not in debug mode and can't connect to hardware
                return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Could not open camera.")
        exit()

    # Reduce resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Track processed faces to avoid duplicate captures
    processed_faces = set()

    # FPS control
    prev_time = 0
    frame_rate = 15  # Target FPS

    # Notification cooldown for unknown faces (seconds)
    notification_cooldown = 30
    last_notification_time = 0
    last_email_time = 0  # Track last email sent time
    
    # Setup health check thread
    stop_event = threading.Event()
    health_thread = threading.Thread(target=health_check_thread, args=(stop_event,), daemon=True)
    health_thread.start()
    
    # Send startup notification
    if mqtt_success:
        send_mqtt_notification({
            "type": "system_status",
            "title": "FaceBot System Active",
            "body": f"Face recognition and tracking is now active (Max {MAX_FACES_FOR_TRACKING} face tracking)",
            "timestamp": datetime.now().isoformat()
        })

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

            # Check for multiple faces and determine if tracking should be enabled
            should_track, primary_face_index = check_multiple_faces(faces, probs)

            if faces is not None:
                valid_faces_processed = 0
                
                # In the main loop where faces are detected, modify the cropping:
                EXPAND_FACTOR = 0.2

                for i, box in enumerate(faces):
                    if probs[i] < 0.90:
                        continue
                        
                    x1, y1, x2, y2 = [int(coord * 2) for coord in box]  # Scale back
                    
                    # Calculate expanded bounding box
                    width = x2 - x1
                    height = y2 - y1
                    expand_w = int(width * EXPAND_FACTOR)
                    expand_h = int(height * EXPAND_FACTOR)
                    
                    # Apply expansion while staying within frame bounds
                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(frame_width, x2 + expand_w)
                    y2 = min(frame_height, y2 + expand_h)
                    
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

                        # In the unknown face handling section:
                        if name == "Unknown" and face_hash not in processed_faces:
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            filename = f"Unknown_{timestamp}.jpg"
                            filepath = os.path.join(SAVED_UNKNOWN_DIR, filename)
                            
                            # Save the larger cropped face
                            cropped_face = frame[y1:y2, x1:x2]
                            
                            # Ensure minimum size (e.g., 300x300 pixels)
                            min_size = 300
                            if cropped_face.shape[0] < min_size or cropped_face.shape[1] < min_size:
                                # Calculate scaling factor
                                scale = max(min_size / cropped_face.shape[0], min_size / cropped_face.shape[1])
                                new_size = (int(cropped_face.shape[1] * scale), int(cropped_face.shape[0] * scale))
                                cropped_face = cv2.resize(cropped_face, new_size, interpolation=cv2.INTER_LINEAR)
                            
                            cv2.imwrite(filepath, cropped_face)
                            
                            if save_unknown_person(cropped_face, face_embedding, cursor):
                                conn.commit()
                                
                                # Check if enough time has passed since last email
                                if current_time - last_email_time >= EMAIL_COOLDOWN:
                                    send_email_async(cropped_face)
                                    last_email_time = current_time
                                    logger.info(f"Email sent for unknown face detection (cooldown: {EMAIL_COOLDOWN}s)")
                                
                                # Send MQTT notification with cooldown
                                if current_time - last_notification_time > notification_cooldown:
                                    if mqtt_success:
                                        success = send_unknown_face_notification(cropped_face)
                                        logger.info(f"MQTT unknown face notification sent: {success}")
                                    last_notification_time = current_time
                            
                            processed_faces.add(face_hash)

                        # Draw bounding box with different colors based on tracking status
                        if valid_faces_processed > MAX_FACES_FOR_TRACKING:
                            # Multiple faces - use yellow for warning
                            color = (0, 255, 255)  # Yellow
                            status_text = f"{name} (TRACKING PAUSED)"
                        elif name != "Unknown":
                            color = (0, 255, 0)  # Green for known
                            status_text = name
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            status_text = name
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, 
                            status_text, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            color, 
                            2
                        )

                        # Face following logic - only for the primary face when tracking is allowed
                        if should_track and i == primary_face_index:
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

                    except Exception as e:
                        logger.error(f"Face processing error: {e}")

                # If tracking is paused due to multiple faces, return to center position
                if not should_track and valid_faces_processed > MAX_FACES_FOR_TRACKING:
                    center_angles = [x_mid, y_mid, 130, 0]
                    if center_angles != prev_servo_angles:
                        if not DEBUG_MODE:
                            send_servo_command(center_angles, ser)
                        prev_servo_angles = center_angles.copy()
                        logger.debug("Robot returned to center position due to multiple faces")

            # Add status indicator on frame
            status_color = (0, 255, 0) if not is_tracking_paused else (0, 255, 255)
            status_text = "TRACKING" if not is_tracking_paused else "PAUSED - MULTIPLE FACES"
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            
            # Add face count indicator
            if faces is not None:
                valid_face_count = sum(1 for prob in probs if prob >= 0.90)
                count_text = f"FACES: {valid_face_count}"
                cv2.putText(
                    frame,
                    count_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Shutdown requested... exiting")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        logger.exception("Details:")
    finally:
        # Stop health check thread
        stop_event.set()
        health_thread.join(timeout=1)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        if not DEBUG_MODE and 'ser' in locals():
            ser.close()
        
        # Send shutdown notification
        if mqtt_success and mqtt_client and mqtt_client.is_connected():
            send_mqtt_notification({
                "type": "system_status",
                "title": "FaceBot Shutting Down",
                "body": "The FaceBot system is shutting down",
                "timestamp": datetime.now().isoformat()
            })
            # Update status to offline
            mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps({
                "type": "robot_status",
                "status": "offline", 
                "timestamp": datetime.now().isoformat()
            }))
            # Stop MQTT client
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.info("MQTT client disconnected")

if __name__ == "__main__":
    main()