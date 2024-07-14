from fastapi import FastAPI, UploadFile, File
from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np

app = FastAPI()
model = EfficientNetB0(weights='imagenet')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = image.load_img(file.file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

