
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow as tf

### Triton client ###
client = httpclient.InferenceServerClient(url="localhost:8000")

### Image pre-processing ###
def image_preprocess(img):
   img = np.asarray(img.resize((224, 224)))
   img = np.stack([img])
   img = tf.keras.applications.resnet50.preprocess_input(img)
   return img

### Image to classify ###
img= Image.open('max/examples/inference/resnet50-python-tensorflow/input/leatherback_turtle.jpg')
img = image_preprocess(img)

### Inference request format ###
inputs = httpclient.InferInput("input_1",
                              img.shape,
                              datatype="FP32")
inputs.set_data_from_numpy(img, binary_data=True)

outputs = httpclient.InferRequestedOutput("predictions",                                         binary_data=True,                                         class_count=1000)

### Submit inference request ###
results = client.infer(model_name="resnet50",
                      inputs=[inputs],
                      outputs=[outputs])
inference_output = results.as_numpy('predictions')

### Process request ###
idx = [int(out.decode().split(':')[1]) for out in inference_output]
probs = [float(out.decode().split(':')[0]) for out in inference_output]

### Decoding predictions ###
probs = np.array(probs)[np.argsort(idx)]
print(tf.keras.applications.resnet.decode_predictions(np.expand_dims(probs, axis=0), top=5))