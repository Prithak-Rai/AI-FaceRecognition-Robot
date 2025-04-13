import cv2
import torch
import time
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Email settings
    SENDER_EMAIL = "prithak.khamtu@gmail.com"
    SENDER_PASSWORD = "paykcwhdbymsukrk"  # Use app-specific password
    RECEIVER_EMAIL = "prithakhamtu@gmail.com"
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    
    # Face recognition settings
    DATA_DIR = "Data"
    UNKNOWN_DIR = "Unknown"
    FACE_DETECTION_THRESHOLD = 0.9
    FACE_MATCH_THRESHOLD = 0.6
    DETECTION_INTERVAL = 10  # Increased for Raspberry Pi performance
    EMAIL_COOLDOWN = 60  # seconds
    
    # Camera settings optimized for Raspberry Pi
    CAMERA_HEIGHT = 240
    CAMERA_WIDTH = 320
    CAMERA_FPS = 15

class FaceRecognitionSystem:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on device: {self.device}")
        self._setup_directories()
        self._initialize_models()
        self._load_known_faces()
        self._initialize_camera()
        self.last_unknown_time = 0
        
    def _setup_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.config.UNKNOWN_DIR, exist_ok=True)
        if not os.path.exists(self.config.DATA_DIR):
            logger.error(f"Data directory '{self.config.DATA_DIR}' not found!")
            raise FileNotFoundError(f"Data directory '{self.config.DATA_DIR}' not found")

    def _initialize_models(self):
        """Initialize face detection and recognition models"""
        logger.info("Initializing models...")
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
            select_largest=False,  # Better for Raspberry Pi performance
            min_face_size=60  # Helps with detection on smaller images
        )
        
        # Initialize FaceNet for face recognition
        self.resnet = InceptionResnetV1(
            pretrained='vggface2',
            classify=False
        ).eval().to(self.device)
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _initialize_camera(self):
        """Initialize camera with error handling"""
        logger.info("Initializing camera...")
        
        # Try different camera indexes
        for i in range(4):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                logger.info(f"Found camera at index {i}")
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
                # Test frame capture
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"Camera at index {i} opened but failed to capture frame")
                    self.cap.release()
                    continue
                return
        
        logger.error("Could not open any working camera!")
        raise RuntimeError("Camera initialization failed")

    def _load_known_faces(self):
        """Load known face embeddings from the data directory"""
        logger.info("Loading known faces...")
        self.known_face_embeddings = []
        self.known_face_names = []
        
        for person_name in os.listdir(self.config.DATA_DIR):
            person_dir = os.path.join(self.config.DATA_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                    
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        continue
                        
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    boxes, _ = self.mtcnn.detect(rgb_image)
                    
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            face = rgb_image[y1:y2, x1:x2]
                            
                            if face.size == 0:
                                continue
                                
                            face_tensor = self.preprocess(face).unsqueeze(0).to(self.device)
                            embedding = self.resnet(face_tensor).detach().cpu().numpy()
                            
                            self.known_face_embeddings.append(embedding)
                            self.known_face_names.append(person_name)
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
        
        if not self.known_face_embeddings:
            logger.warning("No known faces loaded!")
        else:
            logger.info(f"Loaded {len(self.known_face_names)} known faces")

    def _send_email_alert(self, image_path):
        """Send email notification with attached image"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.SENDER_EMAIL
            msg['To'] = self.config.RECEIVER_EMAIL
            msg['Subject'] = "Security Alert: Unknown Person Detected"
            
            body = f"""
            An unknown person was detected by your security system.
            
            Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Image saved at: {image_path}
            """
            msg.attach(MIMEText(body, 'plain'))
            
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read(), name=os.path.basename(image_path))
                msg.attach(img)
            
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.SENDER_EMAIL, self.config.SENDER_PASSWORD)
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def _process_frame(self, frame):
        """Process a single frame for face recognition"""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning("Received empty frame")
                return frame, False
                
            # Resize for faster processing
            small_frame = cv2.resize(frame, (160, 120))  # Smaller size for Raspberry Pi
            
            # Convert to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(rgb_small_frame)
            
            if boxes is None:
                return frame, False
                
            # Scale boxes back to original frame size
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
            
            found_unknown = False
            
            for i, box in enumerate(boxes):
                if probs[i] < self.config.FACE_DETECTION_THRESHOLD:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                    
                # Get face embedding
                face_tensor = self.preprocess(face).unsqueeze(0).to(self.device)
                embedding = self.resnet(face_tensor).detach().cpu().numpy()
                
                # Compare with known faces
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                min_dist = float('inf')
                
                for known_emb, known_name in zip(self.known_face_embeddings, self.known_face_names):
                    dist = np.linalg.norm(embedding - known_emb)
                    if dist < self.config.FACE_MATCH_THRESHOLD and dist < min_dist:
                        min_dist = dist
                        name = known_name
                        color = (0, 255, 0)  # Green for known
                
                # Handle unknown faces
                if name == "Unknown":
                    found_unknown = True
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"unknown_{timestamp}.jpg"
                    img_path = os.path.join(self.config.UNKNOWN_DIR, filename)
                    
                    # Save image and send alert (with cooldown)
                    current_time = time.time()
                    if current_time - self.last_unknown_time > self.config.EMAIL_COOLDOWN:
                        cv2.imwrite(img_path, frame)
                        self._send_email_alert(img_path)
                        self.last_unknown_time = current_time
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, found_unknown
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, False

    def run(self):
        """Main execution loop"""
        logger.info("Starting face recognition system")
        fps_counter = 0
        fps = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)  # Prevent busy waiting
                    continue
                
                # Process every nth frame (for performance)
                if fps_counter % self.config.DETECTION_INTERVAL == 0:
                    processed_frame, _ = self._process_frame(frame)
                else:
                    processed_frame = frame
                
                # Calculate and display FPS
                fps_counter += 1
                if time.time() - start_time >= 1:
                    fps = fps_counter
                    fps_counter = 0
                    start_time = time.time()
                
                cv2.putText(processed_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Face Recognition", processed_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Shutting down by user request")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("System shutdown complete")

if __name__ == "__main__":
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)