from keras.utils import img_to_array, load_img
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from IPython.display import display, Javascript
import google
#from google.colab.output import eval_js
from js2py import eval_js
from base64 import b64decode



model =  load_model('./pothole_final.h5')



def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename






def prepare_img_128(img_path):
    # urllib.request.urlretrieve(img_path, 'save.jpg')
    img = load_img(img_path, target_size=(128, 128))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) / 255
    return x

def predict(img_path):
  img_path = prepare_img_128(img_path)
  pred = model.predict(img_path)
  pred_labels = np.argmax(pred, axis=1)
  print(pred_labels)
  d = {0: 'normal', 1: 'potholes'}
  for key in d.keys():
      if pred_labels[0] == key:
          # print("Result:{} ".format(d[key]))
          return d[key]
  # print("Severity assessment complete.")


  image_file = take_photo()
  predict(image_file)