
import joblib
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
model = joblib.load(MODEL_PATH)




def model_predict(img_path, model):
    # print(img_path)
    # img = image.load_img(img_path, target_size=(200, 200))

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # ## Scaling
    # x=x/255
    img = cv2.imread(img_path, 0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255

    # x = np.expand_dims(img1, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(img1)
    # preds=np.argmax(preds, axis=1)
    print(preds)

    if preds[0] == 1:
        return "Positive Tumor"
    else:
        return "No Tumor"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
