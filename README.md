# Live Face Mask Detection

This project is a real-time computer vision application that opens the webcam and detects whether a person is wearing a face mask or not. It is a lightweight, terminal-based Python application that runs locally on a laptop or desktop without a web interface.

The system uses a two-stage pipeline:

1. A face detector finds faces in each camera frame.
2. A mask classifier decides whether each detected face is `Mask` or `No Mask`.

The result is drawn directly on the live video stream with green boxes for masked faces and red boxes for unmasked faces.

## Repository structure

```text
Face-Mask-Detection-master/
├── detect_mask_video.py         # Main live webcam inference script
├── requirements.txt             # Python dependencies
├── mask_detector.h5             # Trained mask-classification model
├── face_detector/
│   ├── deploy.prototxt          # SSD face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel # Pretrained face detection weights
└── README.md                    # Project documentation
```

## What this project does

The app is designed for real-time safety monitoring and simple mask classification use cases.

It takes a frame from the webcam, detects faces, extracts each face as a cropped image, and sends the cropped face to a deep learning model. The model outputs probabilities such as:

- `mask_probability`
- `no_mask_probability`

The script then compares them and decides the class label:

```python
label = "Mask" if mask_probability > no_mask_probability else "No Mask"
```

This is a classic single-camera inferencing pipeline used in many security, workplace, and public-health monitoring systems.

## Architectural overview

The application architecture is intentionally simple and fast.

### 1. Face detection module

The app uses OpenCV's DNN (Deep Neural Network) module and a pretrained Caffe SSD face detector:

- `face_detector/deploy.prototxt`
- `face_detector/res10_300x300_ssd_iter_140000.caffemodel`

This network scans a frame and returns candidate face boxes with confidence scores. It does not try to classify whether a face is masked—it only finds faces.

### 2. Mask classification module

The mask classifier is loaded from:

- `mask_detector.h5`

This is a Keras/TensorFlow model representing a binary classifier. It receives each face crop resized to 224x224 and returns a 2-value probability vector:

```python
[mask_probability, no_mask_probability]
```

### 3. Real-time video loop

The main loop captures frame by frame from the webcam, runs detection, draws bounding boxes, and updates the window until the user presses `q`.

This is common for embedded vision systems where the system needs to make decisions in milliseconds rather than from static image processing.

## Why this architecture

### Why use two models instead of one big model?

It is easier and more reliable to separate responsibilities:

- Face detection finds where faces are.
- Mask classification decides the mask state.

This makes the system modular, faster to debug, and easier to replace components later.

### Why SSD for face detection?

SSD is a compact, real-time object detector that has good speed/accuracy trade-offs for frontal face detection. It works well for webcam inference on normal desktop hardware.

Alternatives:

- Haar cascades: older, less robust in varied lighting and angles
- YOLO: faster in some setups but more complex to integrate for a simple local app
- MTCNN: very accurate but heavier and slower for a real-time webcam loop

We chose SSD because it is balanced, mature, and works well with OpenCV.

### Why MobileNetV2 for the mask classifier?

The code imports:

```python
from tf_keras.applications.mobilenet_v2 import preprocess_input
```

This suggests the binary mask model is built around MobileNetV2-style preprocessing and likely uses MobileNetV2 as a feature extractor or a transfer-learning base.

Why this is a good choice:

- Pretrained on ImageNet, so it understands visual features quickly
- Small enough for efficient inference on CPU/GPU
- Good trade-off between speed and accuracy
- Commonly used in lightweight mobile and edge deployments

Alternatives:

- VGG16: larger and slower
- ResNet50: stronger but heavier
- Custom CNN from scratch: possible but requires much more training data and time

The chosen approach is the practical one for this project: use transfer learning and keep the system small and fast.

## End-to-end runtime flow

The real runtime flow of the script is:

1. Parse command-line arguments.
2. Load the face detector and mask classifier.
3. Open the webcam.
4. Read a frame.
5. Convert the frame into a blob for the face network.
6. Run SSD on the frame to detect faces.
7. For each face:
   - crop the face
   - convert BGR to RGB
   - resize to 224x224
   - apply MobileNetV2 preprocessing
   - pass to the binary classifier
8. Compare the two probabilities.
9. Draw a rectangle and label on the original frame.
10. Display the live video and repeat until `q` is pressed.

## Code walkthrough

The main logic is in [detect_mask_video.py](detect_mask_video.py).

### 1. Imports

```python
import argparse
import os

import cv2
import numpy as np
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array
```

Each import has a clear purpose:

- `argparse`: parses command-line flags like `--camera`, `--confidence`, and `--model`
- `os`: joins paths for model files and the face detector directory
- `cv2`: OpenCV for video capture, image processing, drawing, and deep learning inference
- `numpy`: matrix and array operations used heavily in detection and prediction
- `preprocess_input`: normalizes input to the format expected by MobileNetV2-based models
- `load_model`: loads the trained Keras mask detector model
- `img_to_array`: converts an image array into a tensor-like array for model input

### 2. The core function: `detect_and_predict_mask`

```python
def detect_and_predict_mask(frame, face_net, mask_net, confidence):
```

This function does the entire detection/classification pipeline for a single frame.

#### a. Extract the image size

```python
height, width = frame.shape[:2]
```

This gets the frame's height and width in pixels. The face detector needs to know the dimensions so it can convert normalized coordinates back to actual pixel coordinates.

#### b. Create the OpenCV DNN blob

```python
blob = cv2.dnn.blobFromImage(
    frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
)
```

This is a very important step.

- `frame`: the current webcam frame
- `1.0`: scale factor
- `(300, 300)`: target size expected by the SSD network
- `(104, 177, 123)`: mean subtraction values in BGR order

The model was trained expecting BGR images with these mean values removed. This helps normalize the input distribution and improve detection robustness.

#### c. Run the face detector

```python
face_net.setInput(blob)
detections = face_net.forward()
```

`face_net` is the SSD face detection network. We feed in the blob and ask it to detect faces in this camera frame.

The `detections` result is a 4D array containing candidate boxes and scores.

#### d. Prepare lists for faces and locations

```python
faces = []
locations = []
```

These hold:

- the face crop images to classify
- the bounding box coordinates for drawing results back on the main image

#### e. Loop through all possible detections

```python
for index in range(detections.shape[2]):
```

The detection tensor includes many candidate boxes. We inspect each one.

```python
score = detections[0, 0, index, 2]
if score <= confidence:
    continue
```

This filters out weak detections. If a detection score is less than or equal to the confidence threshold, it is ignored.

#### f. Convert the detection box into image coordinates

```python
box = detections[0, 0, index, 3:7] * np.array(
    [width, height, width, height]
)
start_x, start_y, end_x, end_y = box.astype("int")
```

OpenCV's face detector outputs normalized values. Multiplying by image width and height converts the box back into real image pixels.

```python
start_x, start_y = max(0, start_x), max(0, start_y)
end_x, end_y = min(width - 1, end_x), min(height - 1, end_y)
```

These lines clamp the coordinates so they stay inside the image boundaries. That prevents negative indices or boxes that extend outside the frame.

#### g. Crop the face from the frame

```python
face = frame[start_y:end_y, start_x:end_x]
if face.size == 0:
    continue
```

This extracts the actual face region from the original frame. If the crop is empty, it is skipped.

#### h. Prepare the face for the mask classifier

```python
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = cv2.resize(face, (224, 224))
faces.append(preprocess_input(img_to_array(face)))
locations.append((start_x, start_y, end_x, end_y))
```

This is the exact preprocessing needed for the trained classification model.

- `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`: OpenCV uses BGR ordering, but the model expects RGB.
- `cv2.resize(face, (224, 224))`: the mask model expects a fixed 224x224 input size.
- `img_to_array(face)`: converts the image to a numeric array.
- `preprocess_input(...)`: applies MobileNetV2-specific normalization.

#### i. Return early if no valid faces were found

```python
if not faces:
    return locations, []
```

If no faces survive the confidence threshold, the function returns empty results quickly.

#### j. Run the mask classifier on all detected faces

```python
predictions = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
return locations, predictions
```

This is the classification step. All detected face crops are batched together and fed into the Keras model.

The model returns one prediction per face, each containing probabilities for the mask classes.

### 3. CLI argument parser

```python
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
```

This gives the app flexibility:

- choose a custom face detector directory
- choose a custom mask model file
- adjust the confidence threshold
- select a different webcam device

### 4. Loading the models

```python
print("[INFO] Loading face detector...")
prototxt_path = os.path.join(args.face, "deploy.prototxt")
weights_path = os.path.join(
    args.face, "res10_300x300_ssd_iter_140000.caffemodel"
)
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading mask detector...")
mask_net = load_model(args.model)
```

This part initializes the two critical networks.

- `readNet` loads the SSD face detector from its protocol buffer and weights.
- `load_model` loads the trained TensorFlow/Keras classifier.

### 5. Webcam setup

```python
camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera {args.camera}. Try --camera 1."
    )
```

This opens the camera tightly on Windows using DirectShow (`CAP_DSHOW`), which is often more stable and flexible than the default backend on Windows systems.

### 6. Main capture loop

```python
while True:
    success, frame = camera.read()
    if not success:
        print("[ERROR] Could not read a frame from the camera.")
        break
```

This keeps reading frames until the camera fails or the user exits.

```python
frame = cv2.resize(frame, (640, 480))
locations, predictions = detect_and_predict_mask(
    frame, face_net, mask_net, args.confidence
)
```

The frame is resized for consistent processing speed. The model still runs on each detected face crop, and the output video is kept at a manageable resolution.

### 7. Decision logic for each detected face

```python
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
```

This is the heart of the classification logic.

- `prediction` is a 2-length array from the model
- `mask_probability` is the probability that the person is wearing a mask
- `no_mask_probability` is the probability they are not
- `label` is chosen by the larger score
- green is used for `Mask` and red for `No Mask`

### 8. Drawing bounding boxes and text

```python
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
```

The app overlays:

- the label (`Mask` or `No Mask`)
- confidence percentage
- bounding box around face

The rectangle draws the face location in the current frame, and the text sits just above the box.

### 9. Display and quit behavior

```python
cv2.imshow("Live Face Mask Detection", frame)
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

This opens a window with the annotated webcam stream. Pressing `q` exits the loop.

### 10. Cleanup

```python
finally:
    camera.release()
    cv2.destroyAllWindows()
```

The script releases the camera hardware and closes all OpenCV windows, keeping the environment clean after exit.

## CNN concept: how the mask classifier works internally

A Convolutional Neural Network (CNN) is a type of deep learning model designed specifically for image data. Instead of looking at an image as a flat array of numbers only, a CNN learns patterns like edges, textures, shapes, nose positions, face contours, and mask boundaries.

This project is a perfect example of CNN use: we do not manually hand-code rules like “if the lower face is covered, then mask.” Instead, the model learns visual patterns automatically from training data.

### Why CNNs are used for images

Images are not just lists of pixels—they have spatial structure. The pixel at the top-left of a face is related to nearby pixels around it, and those relationships matter. A regular fully connected network would treat the image as a long vector, losing the fact that pixels are spatially organized.

CNNs preserve this structure by using local filters that scan small windows across the image.

### The basic idea

A CNN learns small feature detectors, also called kernels or filters. These filters slide across the image and compute how strongly each local region matches a pattern.

For example, one filter may detect vertical edges, another may detect corners, another may detect skin-like textures, and another may detect the outline of a face. As deeper layers process the output, the network combines these low-level patterns into higher-level concepts like “a person wearing a mask.”

### Convolution operation

In CNNs, the main operation is convolution.

If the image is a 224x224x3 tensor (height, width, channels), a filter might be something like 3x3x3, and it slides over the image one step at a time. At each position, it computes a dot product between the filter weights and the local region of the image.

The result is a new matrix called a feature map.

```text
Image patch          Filter              Output value
[ ... ]             [ ... ]             = activation
```

This process makes the network detect patterns such as:

- edges
- corners
- curves
- textures
- face contours
- mask boundaries

### Why the filter size matters

A small filter like 3x3 is often used because it captures local patterns without being too expensive. The filter is tiny compared to the full image, but by sliding across the image it can detect patterns anywhere in the frame.

This is critical in a face mask detector: a mask may appear in different places depending on the face orientation, lighting, or camera angle, and the CNN must detect it regardless of location.

### Stride and padding

Two important CNN parameters are stride and padding:

- Stride: how far the filter moves each time. A stride of 1 moves one pixel at a time; a stride of 2 moves two pixels at a time.
- Padding: adds border pixels around the image so that spatial dimensions do not shrink too quickly and important edge information is preserved.

Without padding, feature maps become smaller after every layer, which can lose border information. In a mask detector, preserving edges around the face is useful because the edge of the nose, cheeks, chin, and mask boundary all carry signal.

### ReLU activation

After a convolution, a non-linear activation is applied, usually ReLU (Rectified Linear Unit):

```python
output = max(0, input)
```

This introduces non-linearity. Without non-linearity, a deep network would behave like a big linear transformation and would not be able to learn complex visual relationships. ReLU is popular because it is simple, fast, and effective.

In practice, after convolution, the network says:

- if a feature is present, keep it
- if not, ignore it

This allows the model to learn more expressive functions.

### Pooling layers

Pooling layers reduce spatial dimensions while keeping important information. The most common is max pooling.

Example:

```text
2x2 region -> maximum value retained
```

This helps with:

- downsampling the feature maps
- reducing computation
- making the model more robust to small shifts or translations

For example, if a mask edge is slightly shifted by a few pixels, the model may still recognize the pattern.

### Hierarchical learning

CNNs learn features in stages:

1. Early layers detect low-level features such as edges and color gradients.
2. Middle layers combine these into shapes and textures.
3. Deep layers detect higher-level patterns such as eyes, nose, cheeks, lips, face structure, and mask coverage.
4. Final layers combine these into the decision: mask or no mask.

This hierarchical learning is one of the main reasons CNNs work so well on images.

### Flattening and classification head

Once the convolutional layers produce feature maps, they are flattened into a vector. This vector is then passed to fully connected dense layers, which act as a classifier.

For example:

```text
Feature maps -> Flatten -> Dense layer -> Output probabilities
```

The final dense layer usually has a size equal to the number of classes. In binary classification, it often produces a 2-element vector representing the two classes:

- class 0: mask
- class 1: no mask

Then a softmax or sigmoid is used to convert the logits into probabilities.

### Softmax and binary output

In this project, the output is effectively a 2-class probability vector. A common formulation is:

```python
[mask_probability, no_mask_probability]
```

where each value is between 0 and 1 and sums to 1.

The code does this comparison:

```python
label = "Mask" if mask_probability > no_mask_probability else "No Mask"
```

This is the final decision rule. The model is not guessing based on a nonsense hand-coded rule; it is using learned representations from training data.

### Why this project uses a CNN for mask recognition

The mask/no-mask problem is visual and subtle. The difference between a masked and unmasked face is not just one bright pixel or a simple threshold—it depends on contextual patterns such as:

- whether the nose and mouth are visible
- how the face is partially occluded
- whether the mask area covers the lower face
- lighting and shadow conditions
- face alignment and expression

A CNN is especially good at learning these patterns because it uses local structure and progressively builds more abstract features.

### How this applies to this repository

The actual script does not train the CNN. It loads a pretrained model from:

```text
mask_detector.h5
```

The script then does this:

1. detects a face in the webcam frame
2. crops the face
3. resizes the crop to 224x224
4. converts the face to RGB
5. applies MobileNetV2 preprocessing
6. passes the face to the CNN
7. receives a probability vector
8. decides whether the face is masked or not

This is exactly how a CNN is used in practice: the model extracts features, then classifies them.

### Why MobileNetV2 is relevant here

The code imports:

```python
from tf_keras.applications.mobilenet_v2 import preprocess_input
```

This tells us the mask classifier is built around a MobileNetV2-style pipeline, which is itself a CNN architecture designed to be efficient. MobileNetV2 uses depthwise separable convolutions, which reduce the number of parameters and computations while preserving strong feature extraction power.

That makes it ideal for real-time inference on edge devices, limited GPUs, or CPUs.

### Transfer learning in plain English

Instead of building a CNN from scratch and training it on a huge face-mask dataset, the project likely uses transfer learning. That means:

- start with a CNN already trained on a large dataset like ImageNet
- keep its early layers, which already know how to detect general visual patterns
- replace or fine-tune the final layers for mask recognition

This is much faster and more data-efficient than training a full CNN from scratch.

### CNN intuition in one sentence

A CNN works by learning filters that respond to patterns in small parts of an image, combining those patterns from local regions into global visual understanding, and finally deciding whether the image belongs to one class or another.

### In this project, the CNN is doing this

For each detected face, the model is effectively asking:

- Are there visual cues that indicate a face covering the lower half?
- Does the image look like a typical masked face or a typical unmasked face?
- Which class has stronger evidence based on learned patterns from training examples?

The answer becomes the probability vector, which is then converted into a label.

### Why the output is not just a random number

The CNN is trained to minimize a loss function, usually binary cross-entropy for a mask/no-mask decision. During training, the model updates its weights so that images of masked faces produce higher probability for the mask class and images of unmasked faces produce higher probability for the no-mask class.

This is a learned mapping from image pixels to class probabilities, not a predetermined rule.

### A simple visual summary

```text
Webcam frame
   -> Face detection (SSD)
   -> Face crop
   -> Resize to 224x224
   -> CNN feature extraction
        - edges
        - textures
        - face landmarks
        - mask-related patterns
   -> Dense classifier
   -> [mask_prob, no_mask_prob]
   -> Decision: Mask or No Mask
```

This is the exact conceptual pipeline used by the project.

## Model building: how the classifier is usually built

This repository includes the trained model file `mask_detector.h5`, but it does not include the training notebook or script itself. That means the model was likely trained offline in a separate training environment.

The usual pattern for building a model like this is:

1. Collect a dataset of face images labeled as `with_mask` and `without_mask`.
2. Crop or detect faces in each image.
3. Resize each face to a size expected by the model, usually 224x224.
4. Use a pre-trained network such as MobileNetV2 as the base feature extractor.
5. Replace the final classification layer with a binary output head:
   - class 0: mask
   - class 1: no mask
6. Train with binary cross-entropy loss.
7. Save the trained model as `mask_detector.h5`.

Conceptually, the model looks roughly like this:

```text
Input (224x224x3)
  -> MobileNetV2 backbone
  -> Global pooling / feature embedding
  -> Dense layer(s)
  -> Binary softmax output
```

The output is not raw logits; it is usually a probability vector of length 2, for example:

```python
[0.92, 0.08]
```

meaning 92% mask, 8% no mask.

### Why binary classification?

The problem is simple enough to treat as a binary task:

- `Mask`
- `No Mask`

We do not need a multi-class detector for all sorts of face accessories. This reduces complexity and speeds up inference.

### Why not train a full face detector from scratch?

Face detection is a different problem than mask classification. It needs a much larger dataset and more computational effort. In real-world projects, it is standard to reuse a pretrained face detector like the SSD model provided by OpenCV.

## Alternatives considered and why they were not chosen

| Option | Advantage | Drawback | Why not chosen here |
|---|---|---|---|
| Haar cascades | Very lightweight and easy | Weak on varied lighting, partial occlusion, and pose changes | Less accurate for real-world webcam conditions |
| MTCNN | Accurate face detection | Heavier and slower | Not necessary for a simple local app |
| YOLO for face detection | Fast and strong | More complex to configure and manage | SSD is simpler and already integrated with OpenCV |
| Full custom CNN | Complete control | Expensive to train, more data required | Transfer learning is faster and more practical |
| MobileNetV2 + fine-tuning | Fast and robust | Needs data quality and tuning | Best balance for this project |
| Web app using Flask/Streamlit | Convenient UI | Extra complexity and no real-time advantage | CLI app is smaller and more direct |

In short, the project prioritizes simplicity, local execution, and good real-time performance over complicated deployment architecture.

## Requirements and setup

### Python dependencies

The project uses the following packages in [requirements.txt](requirements.txt):

```text
tensorflow>=2.20,<2.21
tf-keras>=2.20,<2.21
opencv-python>=4.10,<5
numpy>=1.26,<2.2
```

### Install steps on Windows

Open PowerShell in the project folder and run:

```powershell
py -3.13 -m venv C:\fmvenv
C:\fmvenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\fmvenv\Scripts\Activate.ps1
```

## Run the application

### Default webcam

```powershell
python detect_mask_video.py
```

### Choose another camera

```powershell
python detect_mask_video.py --camera 1
```

### Adjust confidence threshold

```powershell
python detect_mask_video.py --confidence 0.5
```

The script will show a live video stream with bounding boxes and labels. Press `q` to quit.

## Important implementation notes

### Frame resizing

```python
frame = cv2.resize(frame, (640, 480))
```

The script resizes frames for predictable performance. This is a typical compromise: lower-resolution frames are faster but can lose some tiny details. For normal home webcams, this is a good trade-off.

### Confidence threshold

```python
if score <= confidence:
    continue
```

The threshold is the minimum detection confidence required to accept a face. Setting it too low can produce false positives; setting it too high can miss faces.

### Data type and precision

```python
predictions = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
```

The model expects float32 arrays. This is standard for TensorFlow/Keras inference. Using this dtype helps performance and compatibility.

## Performance characteristics

This system is optimized for local inferencing and uses:

- lightweight face detection
- small classification model
- direct webcam processing
- no heavy browser or server stack

This makes it useful for quick demos, classroom projects, edge-device prototyping, and low-cost safety monitoring systems.

## Limitations

This project is intentionally simple and therefore has some real limitations:

- It assumes one or more faces are in front of the camera.
- Performance varies with webcam quality and lighting.
- The system is trained for a broad mask/no-mask rule, not for nuanced face covering detection.
- It is not a full production-grade surveillance system with logging, GDPR controls, or advanced alerting.

## Future improvements

Possible upgrades include:

- tracking the same face across frames to reduce flicker
- using a more robust detector for side profiles and occlusions
- adding confidence calibration and smoothing over time
- moving to a web dashboard or API service
- saving logs and images for audit or analytics

## Summary

This project is a practical, real-time face mask detection system using:

- OpenCV SSD for face detection
- MobileNetV2-based Keras classifier for mask recognition
- webcam input and live visualization in a single Python script

The key idea is simple but powerful: detect the face first, then classify the face crop. That separation keeps the system efficient, interpretable, and easy to extend.

This repository is best viewed as an inference-first, edge-friendly proof of concept. It demonstrates how modern computer vision and transfer learning can be combined into a small, useful application without needing a complex web or cloud infrastructure.
