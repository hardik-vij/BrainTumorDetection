import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumordet10.h5')

image=cv2.imread('C:\\Users\\abc\\Desktop\\FOML\\venv\\archive (2)\\pred\\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

print(img)

input_img=np.expand_dims(img,axis=0)

result=(model.predict(input_img) > 0.5).astype("int32")
print(result)