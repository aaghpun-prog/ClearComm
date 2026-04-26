from flask import Flask, render_template, request, jsonify
import os
import nltk
from modules.length_control import analyze_length_and_rewrite
from modules.homonym_detector import analyze_homonyms_sbert_pipeline
from modules.info_gap_detector import check_info_gaps

# Set local NLTK data path for the whole application
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

app = Flask(__name__)

# --- Dev: always reload templates on change, never serve stale cache ---
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


@app.after_request
def add_no_cache_headers(response):
    """Prevent browser from caching HTML/JSON during development."""
    if 'text/html' in response.content_type or 'application/json' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/length_control', methods=['POST'])
def length_control():
    data = request.get_json()
    text = data.get('text', '')
    target_len = int(data.get('target', 50))
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    try:
        report = analyze_length_and_rewrite(text, target_len)
        return jsonify(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/homonym_detector', methods=['POST'])
def homonym_detector():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    try:
        report = analyze_homonyms_sbert_pipeline(text)
        return jsonify(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/info_gap', methods=['POST'])
def info_gap():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    try:
        report = check_info_gaps(text)
        return jsonify(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
