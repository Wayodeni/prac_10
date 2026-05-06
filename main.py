from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = './best_classification_model.keras'
METADATA_PATH = APP_DIR / 'best_model_metadata.json'

app = FastAPI(title='Image Classification API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
class_names = metadata['class_names']
target_shape = tuple(metadata['input_shape'])
need_flatten = bool(metadata['need_flatten'])


def preprocess_image(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('Изображение не удалось прочитать')
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if len(target_shape) == 3:
        target_h, target_w, target_c = target_shape
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[-1] == 1 and target_c == 3:
            img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] == 3 and target_c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if target_c == 1 and img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    if need_flatten:
        img = img.reshape((img.shape[0], -1))
    return img


@app.get('/health')
def health():
    return {'status': 'ok', 'model': metadata['model_name']}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image(image_bytes)
    probabilities = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    probabilities_dict = {class_names[i]: float(probabilities[i]) for i in range(min(len(class_names), len(probabilities)))}
    return {'predicted_class': class_names[predicted_idx], 'predicted_index': predicted_idx, 'probabilities': probabilities_dict}
