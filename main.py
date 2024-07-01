import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Load model weights and categories
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]  # ['__background__', 'person', 'bicycle', 'car', ...]

# Preprocessing transformation
img_preprocess = weights.transforms()  # Scales values from 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()  # Setting Model for Evaluation/Prediction
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)  # (3,500,500)
    prediction = model([img_processed])[0]  # (1,3,500,500) -> Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"].tolist()]
    return prediction

def create_image_with_bboxes(img, prediction):  # Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img).permute(2, 0, 1)  # Convert (H,W,C) to (C,H,W)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=2
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    return img_with_bboxes_np

# Streamlit Dashboard
st.title("Object Detector :smile: ")
upload = st.sidebar.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)  # Dictionary
    img_with_bbox = create_image_with_bboxes(np.array(img), prediction)  # (H,W,C)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)
