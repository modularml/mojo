import cv2
import numpy as np
import tritonclient.http as httpclient
import tensorflow as tf
import json
import argparse
import time

width = 1280
height = 720
scale_factor = 0.2
text_color = (0, 0, 255)

### Image pre-processing ###
def image_preprocess(preprocess_fn,img,reps=1):
   img = np.asarray(np.resize(img,(224, 224,3)),dtype=np.float32)
   img = np.stack([img]*reps)
   img = preprocess_fn(img)
   return img

def main(args):
    model_name = args.model
    client = httpclient.InferenceServerClient(url="the-machine.local:8000")

    vidcap = cv2.VideoCapture(1)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if model_name=="resnet50":
        preprocess_fn = tf.keras.applications.resnet.preprocess_input
        decode_fn = tf.keras.applications.resnet.decode_predictions
        input_name = "input_1" # from input metadata
    else:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        decode_fn = tf.keras.applications.efficientnet.decode_predictions
        input_name = "input_2" # from input metadata

    time_inference=[]
    walltime_start=time.time()
    while True:
        _, frame = vidcap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scaled_frame = cv2.resize(rgb_frame, (224,224))
        img = image_preprocess(preprocess_fn,scaled_frame)

        inputs = httpclient.InferInput(input_name, 
                                        img.shape, 
                                        datatype="FP32")
        
        inputs.set_data_from_numpy(img, binary_data=True)

        outputs = httpclient.InferRequestedOutput("predictions", 
                                    binary_data=True, 
                                    class_count=1000)
        
        start_time = time.time()        
        results = client.infer(model_name=model_name, 
                inputs=[inputs], 
                outputs=[outputs])
        time_inference.append(time.time() - start_time)

        inference_output = results.as_numpy('predictions')

        ### Process request ###
        idx = [int(out.decode().split(':')[1]) for out in inference_output]
        probs = [float(out.decode().split(':')[0]) for out in inference_output]

        ### Decoding predictions ###
        probs = np.array(probs)[np.argsort(idx)]
        labels = decode_fn(np.expand_dims(probs, axis=0), top=5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, model_name+": "+labels[0][0][1], (30,60), font, 2, text_color, 3, cv2.LINE_AA)
        cv2.imshow('MAX Serving Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Total inference time:",np.sum(time_inference))
            print("Walltime:",time.time() - walltime_start)
            print("Average inference latency(ms):",np.mean(time_inference) * 1000)
            break

    vidcap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                      type=str, 
                      default='resnet50', 
                      help='Choose: Resnet50 | efficientnet')
    main(parser.parse_args())