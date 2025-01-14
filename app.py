import tensorflow
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('Fire_detection.h5')

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  
    img_array = image.img_to_array(frame)
    img_array = img_array/255
    img_array = img_array.reshape(1,224,224,3)
    return img_array

def detect_fire(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    if predictions[0][0] <= 0.5: return True
    return False

input_video_path = "Fire_timelapse.mp4"  
output_video_path = "output_video.mp4"  

cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if detect_fire(frame):
        cv2.putText(frame, "Fire Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
