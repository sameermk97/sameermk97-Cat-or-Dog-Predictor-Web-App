import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
def get_model():
    global model
    global graph 
    graph = tf.get_default_graph()

   # model = load_model('VGG16_cats_and_dogs.h5')
    model = load_model('dogsandcat_vgg16_model_tl.h5')
    model._make_predict_function()

    print(" * Model loaded!")
    
        

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()
global graph 
graph = tf.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    with graph.as_default():

        prediction = model.predict(processed_image).tolist()
        print(prediction)
        response = {
            'prediction': {
                'cat': prediction[0][0]*100,
                'dog': prediction[0][1]*100
            }
        }
        return jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)
