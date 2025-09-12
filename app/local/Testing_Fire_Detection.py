import numpy as np
import keras
import cv2
import os
# Load once at startup
model = keras.models.load_model("app/local/xception_final.keras")

def analyse_frames(frames):
    """
    frames: list of numpy arrays (BGR images)
    returns: list of predictions (fire=1, no_fire=0)
    """
    fire_detections = []

    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Warning: Frame {i} is None (image not found or unreadable).")
            fire_detections.append(None)
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and normalize
        resized = cv2.resize(rgb_frame, (224, 224))
        x = np.expand_dims(resized / 255.0, axis=0)

        # Predict
        pred = model.predict(x, verbose=0)
        print(f"Frame {i} raw prediction: {pred}")

        # Use threshold for binary classification
        fire_prob = 1 - pred[0][0]  # invert
        fire_detections.append(int(fire_prob > 0.5))

    return fire_detections

# Load frames
image_folder = "app/local/Images"
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")  # add more if needed
frames = []

for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(valid_extensions):
        img_path = os.path.join(image_folder, file_name)
        frames.append(cv2.imread(img_path))

detections = analyse_frames(frames)
print("Final detections:", detections)
