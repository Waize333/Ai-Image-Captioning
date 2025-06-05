import gradio as gr
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_images_from_url(url):
    captions = []
    
    try:
        # Download the page
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_elements = soup.find_all('img')

        for img_element in img_elements:
            img_url = img_element.get('src')

            if not img_url:
                continue

            # Skip icons or SVGs
            if 'svg' in img_url or '1x1' in img_url:
                continue

            # Fix malformed URLs
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith('http://') and not img_url.startswith('https://'):
                continue

            try:
                img_resp = requests.get(img_url, timeout=5)
                raw_image = Image.open(BytesIO(img_resp.content))
                if raw_image.size[0] * raw_image.size[1] < 400:
                    continue
                raw_image = raw_image.convert("RGB")

                inputs = processor(images=raw_image, return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(out[0], skip_special_tokens=True)

                captions.append(f"📷 {img_url}\n📝 {caption}")
            except Exception as e:
                captions.append(f"❌ Failed to process: {img_url}")
                continue

        return "\n\n".join(captions) if captions else "No valid images found on the page."

    except Exception as e:
        return f"Error fetching page: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=caption_images_from_url,
    inputs=gr.Textbox(label="Enter Wikipedia URL", placeholder="https://en.wikipedia.org/wiki/IBM"),
    outputs="text",
    title="Web Image Captioning with BLIP",
    description="Enter a Wikipedia article URL to generate captions for its images using the BLIP model."
)

iface.launch()
