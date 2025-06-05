# AI Image Captioner from URLs and Local Images

This project is an AI-powered image captioning tool that automatically generates textual descriptions for images either scraped from a web page (e.g., Wikipedia articles) or located in a local directory. It uses the [BLIP model](https://huggingface.co/Salesforce/blip-image-captioning-base) from Salesforce, integrated with the Hugging Face Transformers library.

##  Features

- Extracts images from a URL (webpage scraping with BeautifulSoup)
- Supports local image directory captioning
- Generates high-quality captions using a pre-trained BLIP model
- Optionally view results via a Gradio-based UI
- Supports JPEG, PNG formats
- Saves results in `captions.txt` file

##  Model Used

- [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

## 
Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-image-captioner.git
cd ai-image-captioner
