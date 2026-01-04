# This code creates a web interface for two purposes:
# 1. Generate captions for uploaded images using the BLIP image-captioning model.
# 2. Demonstrate a simple number addition function using Gradio.
# Users can interact with the app through a browser, uploading images or entering numbers,
# and the app returns either the generated caption or the sum of the numbers.


# Import Gradio for building web-based user interfaces
import gradio as gr
# Import BLIP processor and model for image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration
# Import PIL for image handling

# Load the pretrained BLIP processor
# The processor handles image preprocessing and tensor conversion
# Prepare images for processing by standardizing format and size.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pretrained BLIP model
# This model generates text captions from images
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

from PIL import Image


def add_numbers(Num1, Num2):
    return Num1 + Num2

def generate_caption(image):
    """
    Generates a caption for a given PIL image using the BLIP model.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        str: Generated image caption
    """
    # Now directly using the PIL Image object
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def caption_image(image):
    """
    Wrapper function for Gradio.
    Takes a PIL Image input and returns a caption.
    Includes basic error handling.
    """
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Create a Gradio interface for image captioning
# Users upload an image and receive a generated caption

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption."
)

# Create another Gradio interface for adding numbers
# This demonstrates how Gradio can handle simple functions
demo = gr.Interface(
    fn=add_numbers, 
    inputs=["number", "number"], # Create two numerical input fields where users can enter numbers
    outputs="number" # Create numerical output fields
)
# Launch the Gradio app
# server_name="0.0.0.0" allows access from other devices on the network
# server_port=7860 specifies the port number
#demo.launch(server_name="0.0.0.0", server_port= 7860,debug=True)
iface.launch(server_name="0.0.0.0", server_port= 7860,debug=True)

