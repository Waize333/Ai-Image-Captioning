import os
import glob
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = "/path/to/your/images"  # <-- Replace with your actual directory
image_exts = ["jpg", "jpeg", "png"]  # Supported file types

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    # Iterate over each image file in the directory
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            try:
                # Load the image
                raw_image = Image.open(img_path).convert('RGB')

                # Skip tiny images
                if raw_image.size[0] * raw_image.size[1] < 400:
                    continue

                # Process the image
                inputs = processor(images=raw_image, return_tensors="pt")

                # Generate a caption
                out = model.generate(**inputs, max_new_tokens=50)

                # Decode the output
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Write image path and caption
                caption_file.write(f"{img_path}: {caption}\n")
                print(f"Captioned: {os.path.basename(img_path)} -> {caption}")

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
