from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'saved_models/cnn_alphanumeric_model_20250601_062953.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class mapping based on your notebook
class_names = []
# Classes 0-9: Digits
class_names.extend([str(i) for i in range(10)])
# Classes 10-35: Uppercase letters A-Z
class_names.extend([chr(ord('A') + i) for i in range(26)])

def preprocess_image(image_data):
    """
    Preprocess the drawn image to match the model's expected input format
    """
    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGBA first, then to grayscale properly
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the white background using alpha channel
        background.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
        image = background
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Find the bounding box of the drawn content
    # Look for pixels that are significantly darker than white
    threshold = 240  # Adjust this if needed
    dark_pixels = image_array < threshold
    
    if np.any(dark_pixels):
        # Find coordinates of dark pixels
        coords = np.column_stack(np.where(dark_pixels))
        
        if len(coords) > 0:
            # Get bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding
            padding = 20
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(image_array.shape[0], y_max + padding)
            x_max = min(image_array.shape[1], x_max + padding)
            
            # Crop to the bounding box
            cropped = image_array[y_min:y_max, x_min:x_max]
            
            # Make it square by padding with white
            h, w = cropped.shape
            size = max(w, h)
            square = np.ones((size, size), dtype=np.uint8) * 255
            
            # Center the cropped image in the square
            start_y = (size - h) // 2
            start_x = (size - w) // 2
            square[start_y:start_y+h, start_x:start_x+w] = cropped
            
            image_array = square
    
    # Resize to 28x28 (same as training data)
    image_array = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors (make background black, text white like EMNIST)
    image_array = 255 - image_array
    
    # Normalize pixel values to [0, 1] range (same as training)
    image_array = image_array.astype('float32') / 255.0
    
    # Add some contrast enhancement
    image_array = np.clip(image_array * 1.2, 0, 1)
    
    # Reshape for CNN input: (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Get the character/digit
        predicted_character = class_names[predicted_class]
        
        # Get top 3 predictions for additional insight
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {
                'character': class_names[i],
                'confidence': float(prediction[0][i])
            }
            for i in top_3_indices
        ]
        
        return jsonify({
            'success': True,
            'prediction': predicted_character,
            'confidence': confidence,
            'top_3': top_3_predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/debug_image', methods=['POST'])
def debug_image():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Convert back to viewable format
        debug_image = (processed_image[0, :, :, 0] * 255).astype(np.uint8)
        
        # Convert to base64 for viewing
        _, buffer = cv2.imencode('.png', debug_image)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'debug_image': f'data:image/png;base64,{debug_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run in production mode when in Docker, debug mode locally
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)