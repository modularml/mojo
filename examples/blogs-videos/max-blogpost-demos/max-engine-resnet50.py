
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image

def load_save_resnet50_model(saved_model_dir = 'resnet50_saved_model'):
   model = ResNet50(weights='imagenet')
   shutil.rmtree(saved_model_dir, ignore_errors=True)
   model.save(saved_model_dir, include_optimizer=False, save_format='tf')
saved_model_dir = 'resnet50_saved_model'
load_save_resnet50_model(saved_model_dir)

#============================================#
### MAX Engine Python API ###
from max import engine
sess = engine.InferenceSession()
model = sess.load('resnet50_saved_model')
#============================================#

def image_preprocess(img, reps=1):
   img = np.asarray(img.resize((224, 224)))
   img = np.stack([img]*reps)
   img = tf.keras.applications.resnet50.preprocess_input(img)
   return img

img= Image.open('max/examples/inference/resnet50-python-tensorflow/input/leatherback_turtle.jpg')
img = image_preprocess(img)

### MAX Engine Python API ###
#============================================#
outputs = model.execute(input_1=img)
#============================================#

probs = np.array(outputs['predictions'][0])
print(tf.keras.applications.resnet.decode_predictions(np.expand_dims(probs, axis=0), top=5))