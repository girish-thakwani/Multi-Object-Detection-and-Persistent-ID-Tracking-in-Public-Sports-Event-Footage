import cv2
from ultralytics import YOLO

def theSkynet(input_video_path, output_video_path):
    """
    Runs YOLOv8 detection and tracking on a video, saving the annotated output.
    """
    # 1. Load the YOLOv8 model. 
    # 'yolov8n.pt' is the nano model (fastest). 
    # For better accuracy with occlusions, you can change this to 'yolov8m.pt' or 'yolov8x.pt'.
    # The model will download automatically on the first run.
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # 2. Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}. Please check the path.")
        return

    # 3. Retrieve video properties to format the output video correctly
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_video_path}")
    print("Tracking subjects and generating annotated frames. This may take a while...")

    # 4. Run the tracking algorithm.
    # - source: the input video
    # - persist=True: forces the model to remember IDs across frames (Multi-Object Tracking)
    # - classes=[0]: filters detections to ONLY include class 0 ('person' in COCO dataset)
    # - stream=True: uses a generator to process one frame at a time (memory efficient)
    # - tracker="bytetrack.yaml": Explicitly uses ByteTrack (highly effective for occlusions)
    results = model.track(
        source=input_video_path, 
        persist=True, 
        classes=[0], 
        stream=True,
        tracker="bytetrack.yaml" 
    )

    # 5. Iterate through the processed frames and write them to the output video
    for result in results:
        # result.plot() automatically draws bounding boxes, labels, and tracking IDs
        annotated_frame = result.plot()
        
        # Write the annotated frame to our output video file
        out.write(annotated_frame)
        
        cv2.imshow("Tracking View", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing interrupted")
            break

    # 6. Clean up resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nSuccess! Annotated video saved to: {output_video_path}")

if __name__ == "__main__":
    # Put your downloaded public sports video in the same folder and update the name here:
    input_file = "vid.mp4" 
    
    # The script will generate this file with bounding boxes and IDs:
    output_file = "output_tracked.mp4"
    
    theSkynet(input_file,output_file)
