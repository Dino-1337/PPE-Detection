import cv2
from models.detection_system import PPEFaceDetectionSystem
from models.violation_logger import ViolationLogger
from ultralytics import YOLO
import config
import time

print("="*60)
print("PPE DETECTION VIDEO TEST - DEBUG MODE")
print("="*60)

# Initialize system with LOWER confidence threshold
violation_logger = ViolationLogger("violations.csv", "static/violations", 5, False)
detector = PPEFaceDetectionSystem(
    config.PPE_MODEL_PATH,
    config.FACE_DB_PATH,
    violation_logger=violation_logger
)

# Load model directly for debug analysis
debug_model = YOLO(config.PPE_MODEL_PATH)

# Process video file
video_path = "ppe_testing_2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"\nVideo Properties:")
print(f"  FPS: {fps:.2f}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {duration:.2f} seconds")
print(f"\nProcessing video... (Press 'q' to quit early)\n")

# Detection statistics
frame_count = 0
detection_stats = {
    'person': 0,
    'hardhat': 0,
    'safety_vest': 0,
    'no_hardhat': 0,
    'no_safety_vest': 0,
    'mask': 0,
    'no_mask': 0
}

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("\n✓ Reached end of video")
        break
    
    frame_count += 1
    
    # Run detection with LOWER confidence threshold for debugging
    results = debug_model.predict(frame, imgsz=640, conf=0.25, verbose=False)
    
    # Analyze detections
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        classes = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        
        # Print detections for this frame
        if frame_count % 30 == 0:  # Print every 30 frames (~1 second)
            print(f"\n--- Frame {frame_count}/{total_frames} ({frame_count/fps:.1f}s) ---")
            for cls, conf in zip(classes, confidences):
                class_name = detector.CLASS_MAP.get(int(cls), 'Unknown')
                print(f"  ✓ {class_name}: {conf:.3f}")
                
                # Update statistics
                if int(cls) == 5:
                    detection_stats['person'] += 1
                elif int(cls) == 0:
                    detection_stats['hardhat'] += 1
                elif int(cls) == 7:
                    detection_stats['safety_vest'] += 1
                elif int(cls) == 2:
                    detection_stats['no_hardhat'] += 1
                elif int(cls) == 4:
                    detection_stats['no_safety_vest'] += 1
                elif int(cls) == 1:
                    detection_stats['mask'] += 1
                elif int(cls) == 3:
                    detection_stats['no_mask'] += 1
    
    # Process frame with detector
    processed = detector.process_frame(frame)
    
    # Add frame counter to display
    cv2.putText(processed, f"Frame: {frame_count}/{total_frames}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(processed, f"Time: {frame_count/fps:.1f}s / {duration:.1f}s", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display result
    cv2.imshow('PPE Detection - Debug Mode', processed)
    
    # IMPORTANT: Proper delay for video playback
    # waitKey(30) = ~30ms delay = ~33 FPS playback
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("\n⚠ User quit early")
        break

elapsed_time = time.time() - start_time

# Print final statistics
print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"Processed {frame_count}/{total_frames} frames in {elapsed_time:.2f}s")
print(f"\nDetection Statistics:")
print(f"  Person detections: {detection_stats['person']}")
print(f"  Hardhat detections: {detection_stats['hardhat']}")
print(f"  Safety Vest detections: {detection_stats['safety_vest']}")
print(f"  NO-Hardhat detections: {detection_stats['no_hardhat']}")
print(f"  NO-Safety Vest detections: {detection_stats['no_safety_vest']}")
print(f"  Mask detections: {detection_stats['mask']}")
print(f"  NO-Mask detections: {detection_stats['no_mask']}")

if detection_stats['safety_vest'] == 0 and detection_stats['hardhat'] == 0:
    print("\n⚠ WARNING: No PPE equipment detected!")
    print("  Possible reasons:")
    print("  1. Confidence threshold too high (try lowering in config.py)")
    print("  2. Objects too small/far from camera")
    print("  3. Model not trained on similar scenarios")
    print("  4. Poor lighting or occlusion")

print("="*60)

cap.release()
cv2.destroyAllWindows()