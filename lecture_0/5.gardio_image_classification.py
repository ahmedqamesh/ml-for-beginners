
# This code sets up an image classification web app using PyTorch and Gradio.
# 1. Loads a pretrained ResNet-18 model from PyTorch Hub for ImageNet classification.
# 2. Defines a prediction function that processes an input image, computes softmax probabilities,
#    and returns the top class confidences with human-readable ImageNet labels.
# 3. Creates a Gradio interface where users can upload an image and see the top 3 predicted classes,
#    with example images provided for testing.


## Step 1: Setting up the image classification model

import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

## Step 2: Defining a predict function
import requests
from PIL import Image
from torchvision import transforms
# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")
def predict(inp):
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences


## Step 3: Creating a Gradio interface
import gradio as gr
gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["../data/lion.jpg", "../data/cheetah.jpg"]).launch()