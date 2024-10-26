from ultralytics import YOLO

# Load a model configuration (Ensure the model file exists)
model = YOLO("yolov8n.yaml")

# Train the model with the specified data and parameters
results = model.train(data="config.yaml", epochs=1, devices='0,1')  # Use 'devices' to specify GPUs
