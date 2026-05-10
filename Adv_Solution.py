import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

def theSkynet(input_video_path, output_video_path):
    print("Loading YOLOv8 model for advanced tracking...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- ENHANCEMENT DATA STRUCTURES ---
    # Dictionary to store the past N center points of each active ID for trajectories
    track_history = defaultdict(lambda: []) 
    # Set to store every unique ID that has ever appeared
    total_unique_ids = set()                

    print("Processing frames and calculating movement statistics...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking (verbose=False keeps the console clean)
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        # Get the base annotated frame (bounding boxes and IDs)
        annotated_frame = results[0].plot()
        current_active_count = 0
        
        # Check if any objects are detected in the current frame
        if results[0].boxes.id is not None:
            # Extract bounding boxes (x-center, y-center, width, height) and IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            current_active_count = len(track_ids)
            
            for box, track_id in zip(boxes, track_ids):
                # Update total cumulative count
                total_unique_ids.add(track_id)
                
                # Calculate center coordinates
                x, y, w, h = box
                center = (float(x), float(y))
                
                # --- 1. TRAJECTORY VISUALIZATION ---
                track_history[track_id].append(center)
                if len(track_history[track_id]) > 30:  # Keep the tail length to 30 frames
                    track_history[track_id].pop(0)

                #print(track_history)
                # Draw the trajectory trail
                points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)
                
                # --- 2. MOVEMENT STATISTICS (PIXEL SPEED) ---
                # Calculate Euclidean distance between the current frame and the previous frame
                if len(track_history[track_id]) >= 2:
                    dx = track_history[track_id][-1][0] - track_history[track_id][-2][0]
                    dy = track_history[track_id][-1][1] - track_history[track_id][-2][1]
                    
                    # Pixels per frame * frames per second = Pixels per second
                    speed_px_per_sec = np.sqrt(dx**2 + dy**2) * fps 
                    
                    # Display speed above the bounding box
                    cv2.putText(annotated_frame, f"{speed_px_per_sec:.0f} px/s", 
                                (int(x - w/2), int(y - h/2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- 3. OBJECT COUNT OVER TIME (UI DASHBOARD) ---
        # Draw a semi-transparent black background for readability
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Add the text metrics
        cv2.putText(annotated_frame, f"Active on Screen: {current_active_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Total Unique Subjects: {len(total_unique_ids)}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(annotated_frame)
        
        cv2.imshow("Advanced Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished! Saved to {output_video_path}")

if __name__ == "__main__":
    input_vid="vid.mp4"
    output_vid="advanced_output.mp4"
    theSkynet(input_vid,output_vid)
