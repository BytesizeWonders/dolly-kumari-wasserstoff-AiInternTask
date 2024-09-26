# dolly-kumari-wasserstoff-AiInternTask
## Overview
This project uses an AI pipeline to segment, recognize, and analyze items in an image. The solution makes use of pre-trained deep learning models to segment and recognize objects, extract text, and summarize data. The findings are given in a table that contains all of the data for each object.

## Project Structure 
1. Data - Upload the image data
2. Model - Implements models for segmentation, identification, text extraction, and summarization,
3. utils- functions for preprocessing, postprocessing, and data mapping.
4. tests- Unit tests for the various stages of the pipeline.

## Step Explanation
### 1. Installing Dependencies
To ensure that the required libraries are available, the project installs the following:
- `torch`: Used for loading and running the neural network models.
- `torchvision`: Provides easy access to pre-trained models.
- `opencv-python`: Used for image loading and processing.

```bash
!pip install torch torchvision opencv-python
```

### 2. Importing Libraries
The following libraries are imported:
- `torch` and `torchvision` for deep learning models.
- `cv2` (OpenCV) for image processing.
- `numpy` for numerical computations.
- `matplotlib.pyplot` for visualization.

```python
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### 3. Uploading an Image
Using the Google Colab `files.upload()` function, an image is uploaded by the user. This image will be used for segmentation.

```python
from google.colab import files
uploaded = files.upload()  # Opens a dialog to upload an image
```

### 4. Loading the Image
The uploaded image is loaded using OpenCV (`cv2.imread`) and then converted from BGR to RGB. The image is also transformed into a tensor format for model input using `torchvision.transforms.ToTensor()`.

```python
def load_image(image_path):
    image = cv2.imread(image_path)  # Load image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transform = torchvision.transforms.ToTensor()  # Convert to tensor
    image_tensor = transform(image_rgb)
    return image, image_tensor
```

### 5. Segmenting the Image using Mask R-CNN
A pre-trained Mask R-CNN model (`maskrcnn_resnet50_fpn`) is loaded and used to segment objects in the image. This model outputs masks, bounding boxes, and labels for each object detected in the image.

```python
def segment_image(image_tensor):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    
    # Perform inference
    with torch.no_grad():
        prediction = model([image_tensor])

    # Extract masks, bounding boxes, and labels from the prediction
    masks = prediction[0]['masks']
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    return masks, boxes, labels
```

### 6. Visualizing the Results
Bounding boxes are drawn around detected objects using OpenCV’s `rectangle` function. The segmented image is then displayed using `matplotlib`.

```python
def visualize(image, boxes):
    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)  # Blue color for the bounding box
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    plt.imshow(image)
    plt.show()
```

### 7. Output
The output of the pipeline is an image with bounding boxes highlighting the detected objects. Further stages will include:
- Identifying objects with models like YOLO or CLIP.
- Extracting text using OCR (e.g., Tesseract or EasyOCR).
- Summarizing object attributes using NLP models.

## Conclusion
This AI pipeline demonstrates the use of Mask R-CNN for object segmentation in images. The results include masks, bounding boxes, and object labels, with plans to extend the pipeline for identification, text extraction, and summarization.

