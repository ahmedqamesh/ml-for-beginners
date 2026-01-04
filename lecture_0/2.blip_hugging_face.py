""" 
Performs automatic image captioning using a pretrained BLIP (Bootstrapping Language-Image Pre-training) model 
from Hugging Face’s transformers library. 
The goal is to generate a natural-language textual description (caption) for a given image.
 """
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Load an image
image = Image.open("../data/avatar_imresizer.jpg")
# Prepare the image
inputs = processor(image, return_tensors="pt")
# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)



