import flask
from flask import Flask, request
import torch
from torchvision import transforms as tf
import msgpack
from PIL import Image
from io import BytesIO

app = Flask(__name__)


from genderfilter import Genderfilter
GF = Genderfilter()

transform = tf.Compose([
    tf.Resize((256,256)), 
    tf.ToTensor(), 
    tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route('/', methods=['GET'])
def health_check():
    return msgpack.packb({"status":"healthy"})

@app.route('/gender_filter', methods = ["POST"])
def inference():
    input = msgpack.unpackb(flask.request.data)
    input_img = Image.open(BytesIO(input['img'])).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0)
    result = GF(input_tensor)
    response =  msgpack.packb({'result':float(result)})
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1012)
    
