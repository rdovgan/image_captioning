from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, MarianMTModel, MarianTokenizer
from PIL import Image
import torch
import requests
from io import BytesIO

app = Flask(__name__)

# Captioning model
caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name)
caption_processor = ViTImageProcessor.from_pretrained(caption_model_name)
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)

# Translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-uk"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
translation_model.to(device)


def generate_caption(image: Image.Image) -> str:
    pixel_values = caption_processor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    return caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def translate_to_uk(text: str) -> str:
    inputs = translation_tokenizer([text], return_tensors="pt", padding=True).to(device)
    translated = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)


@app.route('/caption', methods=['POST'])
def caption_image():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in JSON body"}), 400

    try:
        image_response = requests.get(data['url'])
        image_response.raise_for_status()
        image = Image.open(BytesIO(image_response.content)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

    caption_en = generate_caption(image)
    caption_ua = translate_to_uk(caption_en)

    return jsonify({
        "en": caption_en,
        "ua": caption_ua
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4317)
