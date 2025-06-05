import os
import glob
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_local_images(image_files):
    results = []
    for image_path in image_files:
        try:
            raw_image = Image.open(image_path).convert("RGB")

            # Skip small images
            if raw_image.size[0] * raw_image.size[1] < 400:
                results.append((image_path.name, "⚠️ Skipped (image too small)"))
                continue

            inputs = processor(images=raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            results.append((image_path.name, caption))
        except Exception as e:
            results.append((image_path.name, f"❌ Error: {str(e)}"))
    return "\n\n".join([f"📷 {name}\n📝 {caption}" for name, caption in results])

# Gradio Interface
iface = gr.Interface(
    fn=caption_local_images,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"], label="Upload Image Files", file_count="multiple"),
    outputs="text",
    title="Local Image Captioning",
    description="Upload images from your computer to generate captions using the BLIP model.",
)

iface.launch()
