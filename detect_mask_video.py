import argparse
import os

import cv2
import numpy as np
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array


def detect_and_predict_mask(frame, face_net, mask_net, confidence):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locations = []
    for index in range(detections.shape[2]):
        score = detections[0, 0, index, 2]
        if score <= confidence:
            continue

        box = detections[0, 0, index, 3:7] * np.array(
            [width, height, width, height]
        )
        start_x, start_y, end_x, end_y = box.astype("int")
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(width - 1, end_x), min(height - 1, end_y)
        face = frame[start_y:end_y, start_x:end_x]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        faces.append(preprocess_input(img_to_array(face)))
        locations.append((start_x, start_y, end_x, end_y))

    if not faces:
        return locations, []

    predictions = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
    return locations, predictions


parser = argparse.ArgumentParser(description="Live webcam face-mask detection")
parser.add_argument(
    "--face", default="face_detector", help="face detector model directory"
)
parser.add_argument(
    "--model", default="mask_detector.h5", help="mask classifier model"
)
parser.add_argument(
    "--confidence", type=float, default=0.5,
    help="minimum face-detection confidence"
)
parser.add_argument(
    "--camera", type=int, default=0, help="camera index (default: 0)"
)
args = parser.parse_args()

print("[INFO] Loading face detector...")
prototxt_path = os.path.join(args.face, "deploy.prototxt")
weights_path = os.path.join(
    args.face, "res10_300x300_ssd_iter_140000.caffemodel"
)
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading mask detector...")
mask_net = load_model(args.model)

camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera {args.camera}. Try --camera 1."
    )

print("[INFO] Camera started. Press 'q' in the camera window to quit.")
try:
    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Could not read a frame from the camera.")
            break

        frame = cv2.resize(frame, (640, 480))
        locations, predictions = detect_and_predict_mask(
            frame, face_net, mask_net, args.confidence
        )

        for (start_x, start_y, end_x, end_y), prediction in zip(
            locations, predictions
        ):
            mask_probability, no_mask_probability = prediction
            label = (
                "Mask"
                if mask_probability > no_mask_probability
                else "No Mask"
            )
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            text = "{}: {:.2f}%".format(
                label, max(mask_probability, no_mask_probability) * 100
            )
            cv2.putText(
                frame, text, (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
            cv2.rectangle(
                frame, (start_x, start_y), (end_x, end_y), color, 2
            )

        cv2.imshow("Live Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
