import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model
model = load_model(os.path.join(MODEL_FOLDER, 'model.h5'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_pneumonia(img_path):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size to match your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if your model requires it

    # Make prediction
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Pneumonia PNEUMONIA"
    else:
        return "NORMAL"

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
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    app.run(debug=True)





"""
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# Creating a Flask Instance
app = Flask(__name__)


IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'static/uploads'  # Assuming 'static' folder exists for uploaded images
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
model = load_model('model.h5')


def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg / 255.0
    currImg = np.reshape(currImg, (1, 150, 150, 3))
    return currImg


def model_pred(image):
    '''
    Performs predictions based on input image
    '''
    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    # Returns class:
    prediction = model.predict_classes(image)[0]
    '''  # Uncomment if you want probability instead of class
    if prediction == 1:
        return "Pneumonia"
    else:
        return "Normal"'''
    return prediction


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')  # Update with your index.html name


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # Checks if post request was submitted
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # Check if filename is an empty string
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If file is uploaded
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")

            # Preprocessing Image
            image = image_preprocessor(imgPath)

            # Performing Prediction
            pred = model_pred(image)

            # Return the prediction result to the upload.html template
            return render_template('upload.html', name=filename, result=pred)  # Update with your upload.html name
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
"""

"""
def makePredictions(path):
    #Method to predict if image upload is normal or pneumonia
    
    img = Image.open(path) # open the image
    img_d = img.resize((224,224)) #resize the image for model
    
    rgbimg=None #check if image is RGB or not
    
    if len(np.array(img_d).shape)<3:
        rgbimg = Image.new("RGB", img_d.size)
        rgbimg.paste(img_d)
    else:
        rgbimg = img_d
    
    rgbimg = np.array(rgbimg, dtype=np.float64)
    rgbimg = rgbimg.reshape((1,224,224,3))
    predictions = model.predict(rgbimg)
    a = int(np.argmax(predictions))
    
    if a==1:
        a = "Pneumonia"
    else:
        a = "Normal"
        
    return a

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        
        #check if file has been uploaded or not
        if 'img' not in request.files:
           
           return render_template('home.html', filename="unnamed.png", message="Please upload an image")
       
        f = request.files['img'] # take the file from request parameters
       
        #sometimes an empty file is sent after timeout
        if f.filename=='':
            return render_template('home.html', filename="unnamed.png", message="No File Selected!!!")
        
        #check if the file is image or not
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html', filename="unnamed.png", message="Please upload an image with .png or .jpg or.jpeg extension")
        
        #if all cases are passed we save file to upload folder
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_) 
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
            
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        
        return render_template('home.html', filename=f.filename, message=predictions, show=True)
    
    #this method will be called whenever anyone will come to this route
    return render_template('home.html', filename='unnamed.png') # for rendering templates

if __name__ == "__main__":
    app.run(debug=True) # app.run will start web ap http:127
    
"""