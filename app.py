from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import logging

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

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:5000"),
        "X-Title": "Crop Disease Detector"
    }
)

# Global variables for model and translations
model = None
translations = None
class_labels = None

def load_resources():
    """Load ML model, translations, and class labels"""
    global model, translations, class_labels
    
    try:
        # Load TensorFlow model
        model_path = 'model/crop_disease_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}. Using mock predictions.")
        
        # Load translations
        with open('translations/crop_names.json', 'r', encoding='utf-8') as f:
            translations = json.load(f)
        logger.info("Translations loaded successfully")
        
        # Load class labels
        with open('model/class_labels.json', 'r', encoding='utf-8') as f:
            class_labels = json.load(f)
        logger.info("Class labels loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        # Initialize with empty data for development
        translations = {"crops": {}, "diseases": {}, "ui_elements": {}}
        class_labels = {}

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
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_disease(image_path):
    """Predict disease from leaf image"""
    try:
        if model is None:
            # Mock prediction for development
            return {
                'crop': 'tomato',
                'disease': 'early_blight',
                'confidence': 89.5,
                'is_healthy': False,
                'severity': 'moderate'
            }
        
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx]) * 100
        
        # Get disease info from class labels
        disease_info = class_labels.get(str(class_idx), {
            'crop': 'unknown',
            'disease': 'unknown',
            'is_healthy': False
        })
        
        # Determine severity based on confidence and disease type
        severity = 'mild'
        if not disease_info.get('is_healthy', False):
            if confidence > 85:
                severity = 'severe'
            elif confidence > 70:
                severity = 'moderate'
        
        return {
            'crop': disease_info.get('crop', 'unknown'),
            'disease': disease_info.get('disease', 'unknown'),
            'confidence': round(confidence, 2),
            'is_healthy': disease_info.get('is_healthy', False),
            'severity': severity
        }
    
    except Exception as e:
        logger.error(f"Error predicting disease: {str(e)}")
        raise

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
    
    prompt = f"""{language_instructions.get(language, language_instructions['english'])}

**Crop**: {crop_display} ({crop_name})
**Disease Detected**: {disease_display} ({disease_name})
**Farmer Location**: {location or 'India'}

As an agricultural expert, provide clear, actionable treatment advice for this crop disease. Structure your response as follows:

1. **रोग की जानकारी / Disease Overview** (2-3 sentences)
2. **तुरंत करें / Immediate Action** (What to do right now)
3. **जैविक उपचार / Organic Treatment** (3-4 natural methods with dosage)
4. **रासायनिक उपचार / Chemical Treatment** (2-3 pesticides/fungicides with dosage)
5. **रोकथाम / Prevention** (How to prevent in future)
6. **ठीक होने का समय / Recovery Time**

Keep language simple and practical. Use measurements familiar to Indian farmers (liters per acre, grams per liter). Focus on treatments available in rural India."""

    try:
        response = openrouter_client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert agricultural advisor helping Indian farmers. You speak Telugu, Hindi, and English fluently. Provide practical, field-tested advice suitable for small and medium farmers."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error generating treatment advice: {str(e)}")
        # Fallback response
        return f"Treatment advice for {disease_display} in {crop_display} is being prepared. Please try again in a moment."

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
        'translations_loaded': translations is not None
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
