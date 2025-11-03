from flask import Flask, render_template, request, jsonify, session, redirect
import os
from werkzeug.utils import secure_filename
import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Model configuration
MODEL_DIR = 'model'
MODEL_FILENAME = 'plant_disease_model_1_latest.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/spaces/bharath1108/crop-disease-model/resolve/main/plant_disease_model_1_latest.pt"

# Ensure necessary directories exist
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    elif not os.path.isdir(app.config['UPLOAD_FOLDER']):
        os.remove(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
except Exception as e:
    logger.warning(f"Could not create directories: {e}. Will attempt to use them anyway.")

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")

# Global variables for model and translations
model = None
device = None
transforms_pipeline = None
translations = None
class_labels = None
disease_info = None

def download_model_from_huggingface():
    """Download PyTorch model from Hugging Face if not exists locally"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model already exists at {MODEL_PATH} ({file_size / (1024*1024):.2f} MB)")
        return True
    
    logger.info(f"Downloading model from Hugging Face: {MODEL_URL}")
    try:
        # Download with streaming to handle large files
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Model size: {total_size / (1024*1024):.2f} MB")
        
        # Download in chunks with progress
        downloaded_size = 0
        chunk_size = 8192
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (1024 * 1024 * 10) == 0:  # Log every 10MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading model from Hugging Face: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during model download: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

def load_resources():
    """Load PyTorch model, translations, and class labels"""
    global model, device, transforms_pipeline, translations, class_labels, disease_info
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Download model if needed
        model_downloaded = download_model_from_huggingface()
        
        # Load PyTorch model
        if model_downloaded and os.path.exists(MODEL_PATH):
            try:
                logger.info(f"Loading model from {MODEL_PATH}...")
                # Load the entire model (weights + architecture)
                model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
                model.eval()
                
                # Verify model loaded correctly
                if hasattr(model, '__class__'):
                    logger.info(f"PyTorch model loaded successfully: {model.__class__.__name__}")
                else:
                    logger.warning("Model loaded but structure unclear")
                    
            except Exception as e:
                logger.error(f"Error loading PyTorch model: {str(e)}")
                logger.warning("Falling back to mock predictions")
                model = None
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Using mock predictions.")
        
        # Define image transforms (standard ImageNet normalization)
        transforms_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load translations
        with open('translations/crop_names.json', 'r', encoding='utf-8') as f:
            translations = json.load(f)
        logger.info("Translations loaded successfully")
        
        # Load class labels
        with open('model/class_labels.json', 'r', encoding='utf-8') as f:
            class_labels = json.load(f)
        logger.info(f"Class labels loaded: {len(class_labels)} classes")
        
        # Load disease information
        try:
            with open('translations/disease_info.json', 'r', encoding='utf-8') as f:
                disease_info = json.load(f)
            logger.info("Disease information loaded successfully")
        except FileNotFoundError:
            logger.warning("disease_info.json not found. Using basic info.")
            disease_info = {}
        
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        # Initialize with empty data for development
        translations = {"crops": {}, "diseases": {}, "ui_elements": {}}
        class_labels = {}
        disease_info = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_translation(key, lang='english'):
    """Get translation for a given key and language"""
    if not translations:
        return key
    
    parts = key.split('.')
    current = translations
    
    try:
        for part in parts:
            current = current[part]
        return current.get(lang, current.get('english', key))
    except (KeyError, TypeError):
        return key

def preprocess_image(image_path):
    """Preprocess image for PyTorch model prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transforms_pipeline(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_disease(image_path):
    """Predict disease from leaf image using PyTorch model"""
    try:
        if model is None:
            # Mock prediction for development/testing
            logger.warning("Using mock prediction - model not loaded")
            return {
                'crop': 'tomato',
                'disease': 'early_blight',
                'confidence': 89.5,
                'is_healthy': False,
                'severity': 'moderate'
            }
        
        # Preprocess image
        img_tensor = preprocess_image(image_path).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_idx = predicted.item()
            confidence_score = confidence.item() * 100
        
        logger.info(f"Prediction: class {class_idx}, confidence {confidence_score:.2f}%")
        
        # Get disease info from class labels
        disease_info_item = class_labels.get(str(class_idx), {
            'crop': 'unknown',
            'disease': 'unknown',
            'is_healthy': False
        })
        
        # Determine severity based on confidence and disease type
        severity = 'mild'
        if not disease_info_item.get('is_healthy', False):
            if confidence_score > 85:
                severity = 'severe'
            elif confidence_score > 70:
                severity = 'moderate'
        
        return {
            'crop': disease_info_item.get('crop', 'unknown'),
            'disease': disease_info_item.get('disease', 'unknown'),
            'confidence': round(confidence_score, 2),
            'is_healthy': disease_info_item.get('is_healthy', False),
            'severity': severity
        }
    
    except Exception as e:
        logger.error(f"Error predicting disease: {str(e)}")
        raise

def get_disease_info(crop_name, disease_name, language='english'):
    """Get detailed disease information"""
    if not disease_info:
        return None
    
    key = f"{crop_name}_{disease_name}"
    info = disease_info.get(key, {})
    
    if info:
        return info.get(language, info.get('english', None))
    return None

def generate_treatment_advice(crop_name, disease_name, language, location=""):
    """Generate treatment recommendations using OpenRouter LLaMA"""
    
    # Language-specific instructions
    language_instructions = {
        'english': "Provide treatment advice in English.",
        'telugu': "Provide treatment advice in Telugu (తెలుగు). Use Telugu script and simple farming terminology.",
        'hindi': "Provide treatment advice in Hindi (हिंदी). Use Devanagari script and simple farming terminology."
    }
    
    # Get translated crop and disease names
    crop_display = get_translation(f'crops.{crop_name}', language)
    disease_display = get_translation(f'diseases.{disease_name}', language)
    
    # Get additional disease info if available
    extra_info = get_disease_info(crop_name, disease_name, language)
    extra_context = f"\n**Additional Context**: {extra_info}" if extra_info else ""
    
    prompt = f"""{language_instructions.get(language, language_instructions['english'])}

**Crop**: {crop_display} ({crop_name})
**Disease Detected**: {disease_display} ({disease_name})
**Farmer Location**: {location or 'India'}{extra_context}

As an agricultural expert, provide clear, actionable treatment advice for this crop disease. Structure your response as follows:

1. **रोग की जानकारी / Disease Overview** (2-3 sentences)
2. **तुरंत करें / Immediate Action** (What to do right now)
3. **जैविक उपचार / Organic Treatment** (3-4 natural methods with dosage)
4. **रासायनिक उपचार / Chemical Treatment** (2-3 pesticides/fungicides with dosage)
5. **रोकथाम / Prevention** (How to prevent in future)
6. **ठीक होने का समय / Recovery Time**

Keep language simple and practical. Use measurements familiar to Indian farmers (liters per acre, grams per liter). Focus on treatments available in rural India."""

    try:
        # Make request to OpenRouter API using requests
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": APP_URL,
                "X-Title": "Crop Disease Detector"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert agricultural advisor helping Indian farmers. You speak Telugu, Hindi, and English fluently. Provide practical, field-tested advice suitable for small and medium farmers."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            },
            timeout=30  # 30 second timeout
        )
        
        response.raise_for_status()  # Raise exception for bad status codes
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.Timeout:
        logger.error("OpenRouter API request timed out")
        fallback_messages = {
            'english': f"Treatment advice request timed out. Please try again in a moment.",
            'hindi': f"उपचार सलाह अनुरोध समय समाप्त हो गया। कृपया कुछ देर में फिर से प्रयास करें।",
            'telugu': f"చికిత్స సలహా అభ్యర్థన సమయం ముగిసింది. దయచేసి కొద్దిసేపు తర్వాత మళ్లీ ప్రయత్నించండి."
        }
        return fallback_messages.get(language, fallback_messages['english'])
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter API: {str(e)}")
        # Fallback response
        fallback_messages = {
            'english': f"Treatment advice for {disease_display} in {crop_display} is being prepared. Please try again in a moment.",
            'hindi': f"{crop_display} में {disease_display} के लिए उपचार सलाह तैयार की जा रही है। कृपया कुछ देर में फिर से प्रयास करें।",
            'telugu': f"{crop_display}లో {disease_display} కోసం చికిత్స సలహా సిద్ధం చేయబడుతోంది. దయచేసి కొద్దిసేపు తర్వాత మళ్లీ ప్రయత్నించండి."
        }
        return fallback_messages.get(language, fallback_messages['english'])
    
    except Exception as e:
        logger.error(f"Unexpected error generating treatment advice: {str(e)}")
        fallback_messages = {
            'english': f"Unable to generate treatment advice at this time. Please try again later.",
            'hindi': f"इस समय उपचार सलाह तैयार करने में असमर्थ। कृपया बाद में पुन: प्रयास करें।",
            'telugu': f"ఈ సమయంలో చికిత్స సలహాను రూపొందించడం సాధ్యం కాలేదు. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి."
        }
        return fallback_messages.get(language, fallback_messages['english'])

@app.route('/')
def index():
    """Home page with language selection"""
    language = session.get('language', 'english')
    return render_template('index.html', 
                         lang=language, 
                         translations=translations)

@app.route('/set-language/<lang>')
def set_language(lang):
    """Set user's preferred language"""
    if lang in ['english', 'hindi', 'telugu']:
        session['language'] = lang
        return jsonify({'success': True, 'language': lang})
    return jsonify({'success': False, 'error': 'Invalid language'}), 400

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and process leaf image"""
    language = session.get('language', 'english')
    
    if request.method == 'GET':
        return render_template('upload.html', 
                             lang=language, 
                             translations=translations)
    
    # POST request - handle file upload
    try:
        if 'leaf_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['leaf_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        # Predict disease
        prediction = predict_disease(filepath)
        
        # Store prediction in session for result page
        session['prediction'] = prediction
        session['image_filename'] = filename
        
        return jsonify({
            'success': True,
            'redirect': '/result'
        })
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': 'Error processing image. Please try again.'}), 500

@app.route('/result')
def result():
    """Display disease detection results"""
    language = session.get('language', 'english')
    prediction = session.get('prediction')
    image_filename = session.get('image_filename')
    
    if not prediction:
        return redirect('/')
    
    # Generate treatment advice
    treatment = None
    if not prediction.get('is_healthy', False) and prediction['confidence'] > 60:
        location = request.args.get('location', '')
        treatment = generate_treatment_advice(
            crop_name=prediction['crop'],
            disease_name=prediction['disease'],
            language=language,
            location=location
        )
    
    return render_template('result.html',
                         prediction=prediction,
                         treatment=treatment,
                         image_filename=image_filename,
                         lang=language,
                         translations=translations)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for disease prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save and process
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"api_{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        prediction = predict_disease(filepath)
        
        # Generate treatment if requested
        language = request.form.get('language', 'english')
        include_treatment = request.form.get('include_treatment', 'true').lower() == 'true'
        
        response = {
            'success': True,
            'prediction': prediction,
            'image_url': f"/static/uploads/{filename}"
        }
        
        if include_treatment and not prediction.get('is_healthy', False):
            treatment = generate_treatment_advice(
                crop_name=prediction['crop'],
                disease_name=prediction['disease'],
                language=language
            )
            response['treatment'] = treatment
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_exists': os.path.exists(MODEL_PATH),
        'translations_loaded': translations is not None,
        'device': str(device) if device else 'not set',
        'num_classes': len(class_labels) if class_labels else 0
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load resources on startup
    load_resources()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
