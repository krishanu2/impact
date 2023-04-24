import cv2
import numpy as np

# Load the TensorFlow Lite model
net = cv2.dnn.readNetFromTensorflowLite('model.tflite')

# Initialize the camera
camera = cv2.VideoCapture(0)

# Loop through frames from the camera
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Preprocess the image
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (2.0 / 255.0) * input_data - 1.0

    # Run inference on the model
    net.setInput(input_data)
    output_data = net.forward()

    # Postprocess the output
    for detection in output_data[0][0]:
        if detection[2] > 0.5:
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()

