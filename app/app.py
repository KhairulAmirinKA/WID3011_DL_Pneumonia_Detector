import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from keras import utils
import numpy as np
from keras import models

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model
# Define the relative path to the model file
model_path = os.path.join('models', 'best_inception_model.keras')

# Load the model
model = models.load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_pneumonia(img_path):
    # Load the image and preprocess it
    img = utils.load_img(img_path, target_size=(299, 299))  # Adjust size to match your model's input
    img_array = utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if your model requires it

    # Make prediction
    prediction = model.predict(img_array)

    # threshold = 0.5
    if prediction[0][0] > 0.5:

        confidence = prediction[0][0]
        response = {
            "result":"Pneumonia",
            "confidence": confidence
        }
        return response

    else:

        confidence = 1- prediction[0][0]
        response = {
            "result": "Normal",
            "confidence": confidence
        }
        return response


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Step 1: Handle file upload
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Step 1: Render `upload.html` for confirmation
        return render_template(
            'upload.html',
            image_url=url_for('static', filename=f'uploads/{filename}'),
            file_name=filename
        )

    return redirect(url_for('index'))

@app.route('/result', methods=['POST'])
def show_result():
    # Step 2: Process the file and render the result
    file_name = request.form.get('file_name')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    if os.path.exists(file_path):
        prediction = predict_pneumonia(file_path)
        return render_template(
            'result.html',
            image_url=url_for('static', filename=f'uploads/{file_name}'),
            prediction=prediction
        )

    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
