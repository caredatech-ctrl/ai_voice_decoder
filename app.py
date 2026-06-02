import os
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition

app = Flask(__name__)

# This is critical: It allows your GitHub Pages URL to talk to this Python server securely
CORS(app) 

print("Loading AI Model (ECAPA-TDNN)...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
MATCH_THRESHOLD = 0.25 

def download_and_convert(url, output_wav_path):
    """Downloads audio from Supabase URL and converts to standard 16kHz WAV for AI."""
    temp_download_path = output_wav_path + "_raw"
    
    # Download file from Supabase
    response = requests.get(url)
    with open(temp_download_path, 'wb') as f:
        f.write(response.content)

    # Convert to 16kHz Mono
    audio = AudioSegment.from_file(temp_download_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_wav_path, format="wav")
    
    # Clean up raw file
    os.remove(temp_download_path) 

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url1 = data.get('audio_url_1')
    url2 = data.get('audio_url_2')

    if not url1 or not url2:
        return jsonify({"error": "Missing audio URLs."}), 400

    try:
        temp_dir = tempfile.mkdtemp()
        wav1 = os.path.join(temp_dir, "voice1.wav")
        wav2 = os.path.join(temp_dir, "voice2.wav")

        # Process the Supabase URLs
        download_and_convert(url1, wav1)
        download_and_convert(url2, wav2)

        # Run AI Verification
        score, prediction = verification_model.verify_files(wav1, wav2)
        raw_score = score.item()
        
        # Calculate human-readable percentage
        normalized_percentage = max(0.0, min(100.0, ((raw_score + 1) / 2) * 100))
        is_match = raw_score > MATCH_THRESHOLD

        # Clean up temporary files
        os.remove(wav1)
        os.remove(wav2)
        os.rmdir(temp_dir)

        # Send result back to HTML
        return jsonify({
            "match": is_match,
            "percentage": round(normalized_percentage, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Required for production cloud deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
