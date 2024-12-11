from flask import Flask, request, jsonify
import musicgen

app = Flask(__name__)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    music = musicgen.generate(prompt)

    return jsonify({'music': music})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
