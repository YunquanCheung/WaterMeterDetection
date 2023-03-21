from models.experimental import attempt_load
from utils.torch_utils import select_device
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
import json
import numpy as np
from backend.predict import predict
from pathlib import Path

app = Flask(__name__)

with open('./backend/flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)


device = select_device(opt['device'])
model = attempt_load(opt['weights'], map_location=device)  

@app.route('/predict/', methods=['POST'])
def get_prediction():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  
    image = base64.b64decode(base64_str) 
    img = Image.open(io.BytesIO(image)) 
    if (img.mode != 'RGB'):
        img = img.convert("RGB")
    save_path = str(Path(opt['source']) / Path("img4predict.jpg")) 
    img.save(save_path) 
    # img.save("./frontend/static/images/img4predict.jpg")  

    # convert to numpy array.
    img_arr = np.array(img)
    

    results = predict(opt, model, img_arr) 

    return jsonify(results)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    #app.run(debug=True, host='127.0.0.1')
    app.run(debug=False, host='0.0.0.0')



