import base64
from io import BytesIO

import numpy as np
import torch as torch
from PIL import Image
from flask import Flask, request

from model import predict_image

app = Flask(__name__)


@app.get('/')
def index():
    return {'message': 'Ping'}


@app.post('/')
def predict():
    base64_image = request.form['image']
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).resize((256, 256)).rotate(-90)
    image_np = np.array(image)

    try:
        image_np = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
        prediction = predict_image(image_np)
    except Exception as e:
        return {'error': str(e)}

    return {'prediction': prediction}


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0')
