import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load YOLO model with CPU support (without CUDA)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
model.to('cpu')  # Ensure model runs on CPU

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Depth')
ax.set_title("3D Object Mapping")

# Object styles and mappings
object_styles = {
    "bird" : {"color": '#2ff20c', "label": "bird"},
    "balloons" : {"color": '#2d3138', "label": "Ballon"},
    "airplane": {"color": 'r', "label": "Plane"},
    "drone": {"color": 'g', "label": "Drone"},
    "fighter_jet": {"color": 'b', "label": "Fighter Jet"},
    "helicopter" : {"color": '#54140e', "label": "Helicopter"},
}

# Open video capture for real-time detection
video_path = r"C:\Users\Shahin Kaushar\Documents\aeroplane.mp4"  # Corrected path string
cap = cv2.VideoCapture(video_path)

# Define output video parameters
output_path = "output_video_aeroplane.mp4"  # Output file name
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (or use 'MP4V' for .mp4)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Frame rate (default to 30 if unknown)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to estimate depth (z) for demonstration purposes
def estimate_depth(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    return 1000 / max(area, 1)  # Inverse proportion for depth estimation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Fetch detections on CPU in x1, y1, x2, y2, conf, class

    # Clear previous points for fresh update in 3D plot
    ax.cla()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Depth')
    ax.set_title("3D Object Mapping")

    for det in detections:
        class_id = int(det[5])
        confidence = det[4]

        if confidence < 0.5:
            continue  # Filter out low-confidence detections

        # Get bounding box and estimate depth
        x1, y1, x2, y2 = map(int, det[:4])
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        z = estimate_depth(x1, y1, x2, y2)

        # Determine object type and style
        class_name = model.names[class_id]
        if class_name in object_styles:
            style = object_styles[class_name]
            color = style["color"]

            # Plot in 3D space
            ax.scatter(center_x, center_y, z, c=color, label=style["label"], s=50)

            # Display bounding box and label on video feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"{style['label']} | Confidence: {int(confidence * 100)}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the video feed and 3D plot
    cv2.imshow("Object Tracking Feed", frame)
    out.write(frame)  # Write the frame to output video
    plt.pause(0.01)  # Update the plot in real-time

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save and close the output video file
cv2.destroyAllWindows()
plt.close()
