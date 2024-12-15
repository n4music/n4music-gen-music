import logging
import os
from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import scipy.io.wavfile as wavfile
import numpy as np
import requests
from diffusers import DiffusionPipeline
from PIL import Image
import io

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small").to(device)

# Load the diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_lora_weights("CiroN2022/cd-md-music")

API_KEY = ""
HUGGINGFACE = ""

API_URL = "https://api-inference.huggingface.co/models/CiroN2022/cd-md-music"
headers = {"Authorization": "Bearer " + HUGGINGFACE}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

@app.route('/generate_music', methods=['POST'])
def generate_music():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Simulate music generation process
        logging.info("Starting music generation process...")

        # Tokenize the prompt and move to GPU
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate audio waveform
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=1000)

        # Move outputs back to CPU and convert to numpy array
        audio_data = outputs.cpu().numpy()
        sample_rate = 44100  # Standard audio sample rate

        if audio_data is None:
            return jsonify({'error': 'Model output missing waveform'}), 500

        music_file_path = "generated_music.wav"
        wavfile.write(music_file_path, sample_rate, audio_data)  # Write WAV file
        logging.info("Music generation completed. File saved at %s", music_file_path)
    except Exception as e:
        logging.error("Error during generation: %s", str(e))
        return jsonify({'error': f'Error during generation: {str(e)}'}), 500

    try:
        logging.info("Sending generated music file...")
        return send_file(music_file_path, as_attachment=True)
    finally:
        if os.path.exists(music_file_path):
            os.remove(music_file_path)  # Clean up file after sending
            logging.info("Cleaned up generated music file.")

@app.route('/generate_song_name', methods=['POST'])
def generate_song_name():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    description = data.get('description', '')

    if not description:
        return jsonify({'error': 'Description is required'}), 400

    try:
        logging.info("Starting song name generation process...")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"\"Dựa trên nội dung sau, hãy tạo một tên bài hát tiếng Việt phù hợp và độc đáo:\n\nMô tả nội dung bài hát: {description}\".\n\nChỉ trả ra tên bài hát, không thêm bất kỳ thông tin nào khác.\""
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "text/plain"
            }
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent?key={API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=payload
        )

        if response.status_code != 200:
            logging.error("Error from external API: %s", response.text)
            return jsonify({'error': 'Error from external API'}), 500
        response_data = response.json()
        logging.info("Response from external API: %s", response_data)

        song_name = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        logging.info("Song name generation completed: %s", song_name)
        return jsonify({'song_name': song_name})
    except Exception as e:
        logging.error("Error during song name generation: %s", str(e))
        return jsonify({'error': f'Error during song name generation: {str(e)}'}), 500

@app.route('/generate_prompt_suggestion', methods=['POST'])
def generate_prompt_suggestion():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    context = data.get('context', '')

    if not context:
        return jsonify({'error': 'Context is required'}), 400

    try:
        logging.info("Starting prompt suggestion generation process...")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"\"Dựa trên nội dung sau, hãy tạo một gợi ý mô tả hoặc prompt hoàn chỉnh để viết một prompt hoàn chỉnh truyền vào musicgen:\n\nNội dung: {context}\".\n\nChỉ trả ra gợi ý, không thêm bất kỳ thông tin nào khác.\""
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "text/plain"
            }
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent?key={API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=payload
        )

        if response.status_code != 200:
            logging.error("Error from external API: %s", response.text)
            return jsonify({'error': 'Error from external API'}), 500
        response_data = response.json()
        logging.info("Response from external API: %s", response_data)

        suggestion = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        logging.info("Prompt suggestion generation completed: %s", suggestion)
        return jsonify({'suggestion': suggestion})
    except Exception as e:
        logging.error("Error during prompt suggestion generation: %s", str(e))
        return jsonify({'error': f'Error during prompt suggestion generation: {str(e)}'}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    context = data.get('context', '')

    if not context:
        return jsonify({'error': 'Context is required'}), 400

    try:
        logging.info("Starting image generation process...")

        # Generate image based on context using Hugging Face API
        image_bytes = query({"inputs": context})

        # Access the image with PIL.Image
        image = Image.open(io.BytesIO(image_bytes))

        # Save image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        logging.info("Image generation completed.")
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
    except Exception as e:
        logging.error("Error during image generation: %s", str(e))
        return jsonify({'error': f'Error during image generation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
