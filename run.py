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
from kafka import KafkaConsumer, KafkaProducer
import threading
import json
from flask_cors import CORS

app = Flask(__name__)

# Update CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3001", "http://localhost:3000"],  # Add your frontend origins
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3001')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

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
MEDIA_FILE = "http://localhost:3076/files/upload"

headers = {"Authorization": "Bearer " + HUGGINGFACE}

# Kafka configuration
KAFKA_BROKER_URL = 'localhost:9092'
KAFKA_CONSUMER_GROUP = 'music_generation_group'
KAFKA_TOPICS = ['generate_music', 'generate_song_name', 'generate_image']

# Initialize Kafka consumer
consumer = KafkaConsumer(
    *KAFKA_TOPICS,
    bootstrap_servers=KAFKA_BROKER_URL,
    group_id=KAFKA_CONSUMER_GROUP,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER_URL,
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

@app.route('/generate_music', methods=['POST'])
def generate_music_route():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    prompt = data.get('context', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    return generate_music(prompt)

def generate_music(prompt):
    try:
        # Simulate music generation process
        logging.info("Starting music generation process...")

        # Tokenize the prompt and move to GPU
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate audio waveform
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=100)

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
        logging.info("Sending generated music file to media file API...")
        with open(music_file_path, 'rb') as f:
            files = {'file': ('generated_music.wav', f, 'audio/wav')}
            response = requests.post(MEDIA_FILE, files=files)

        if response.status_code != 201:
            logging.error("Error from media file API: %s", response.text)
            return jsonify({'error': 'Error from media file API'}), 500

        logging.info("Music file successfully uploaded to media file API.")
        return jsonify(response.json())
    finally:
        if os.path.exists(music_file_path):
            os.remove(music_file_path)  # Clean up file after sending
            logging.info("Cleaned up generated music file.")

@app.route('/generate_song_name', methods=['POST'])
def generate_song_name_route():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    description = data.get('context', '')

    if not description:
        return jsonify({'error': 'Description is required'}), 400

    return generate_song_name(description)

def generate_song_name(description):
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
def generate_image_route():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    context = data.get('context', '')

    if not context:
        return jsonify({'error': 'Context is required'}), 400

    return generate_image(context)
def generate_image(context):
    try:
        logging.info("Starting image generation process...")

        # Generate image based on context using Hugging Face API
        image_bytes = query({"inputs": context})

        # Access the image with PIL.Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        logging.info("Image generation completed.")

        # Send the generated image to the media file API
        files = {'file': ('generated_image.png', img_byte_arr, 'image/png')}
        response = requests.post(MEDIA_FILE, files=files)

        if response.status_code != 201:
            logging.error("Error from media file API: %s", response.text)
            return jsonify({'error': 'Error from media file API'}), 500

        logging.info("Image successfully uploaded to media file API.")
        return jsonify(response.json())
    except Exception as e:
        logging.error("Error during image generation: %s", str(e))
        return jsonify({'error': f'Error during image generation: {str(e)}'}), 500

@app.route('/gen-ai', methods=['POST'])
def gen_ai_route():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    context = data.get('context', '')
    task = data.get('task', '')
    handlers = {
        'generate_music': handle_generate_music,
        'generate_song_name': handle_generate_song_name,
        'generate_image': handle_generate_image
    }

    if not context:
        return jsonify({'error': 'Context is required'}), 400
    if task not in handlers:
        return jsonify({'error': 'Invalid task'}), 400

    threading.Thread(target=handlers[task], args=(data,)).start()
    return jsonify({'status': 'Processing started'}), 200
    # Default to running all handlers if no specific task is provided
@app.route('/gen-ai-all', methods=['POST'])
def gen_ai_all_route():
    if not request.json:
        return jsonify({'error': 'Invalid JSON input'}), 400

    data = request.json
    context = data.get('context', '')

    if not context:
        return jsonify({'error': 'Context is required'}), 400

    # Start threads for all handlers
    threading.Thread(target=handle_generate_music, args=(data,)).start()
    threading.Thread(target=handle_generate_song_name, args=(data,)).start()
    threading.Thread(target=handle_generate_image, args=(data,)).start()

    return jsonify({'message': 'Processing started for all tasks'}), 200

def create_response(data, status=200):
    """Create a standardized response dictionary"""
    return {
        'status': status,
        'data': data
    }

def response_to_dict(response):
    """Convert Flask response to dictionary"""
    if hasattr(response, 'get_json'):
        # If it's a Flask response object
        return {
            'status_code': response.status_code,
            'data': response.get_json()
        }
    elif isinstance(response, tuple) and len(response) == 2:
        # If it's a tuple of (response, status_code)
        data, status_code = response
        if hasattr(data, 'get_json'):
            return {
                'status_code': status_code,
                'data': data.get_json()
            }
    return response  # Return as-is if not a Flask response

def handle_generate_music(data):
    with app.app_context():
        try:
            context = data.get('context', '')
            if not context:
                result = create_response({'error': 'Context is required'}, 400)
            else:
                result = response_to_dict(generate_music(context))
            producer.send('generate_music_result', value=result)
        except Exception as e:
            logging.error(f"Error in music generation thread: {str(e)}")
            producer.send('generate_music_result', value=create_response(
                {'error': str(e)}, 500))

def handle_generate_song_name(data):
    with app.app_context():
        try:
            description = data.get('context', '')
            if not description:
                result = create_response({'error': 'Description is required'}, 400)
            else:
                result = response_to_dict(generate_song_name(description))
            producer.send('generate_song_name_result', value=result)
        except Exception as e:
            logging.error(f"Error in song name generation thread: {str(e)}")
            producer.send('generate_song_name_result', value=create_response(
                {'error': str(e)}, 500))

def handle_generate_image(data):
    with app.app_context():
        try:
            context = data.get('context', '')
            if not context:
                result = create_response({'error': 'Context is required'}, 400)
            else:
                result = response_to_dict(generate_image(context))
            producer.send('generate_image_result', value=result)
        except Exception as e:
            logging.error(f"Error in image generation thread: {str(e)}")
            producer.send('generate_image_result', value=create_response(
                {'error': str(e)}, 500))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
