from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task', methods=['POST'])

def predict():
    """
    Ini Adalah Endpoint Untuk Memprediksi Instagram Comment
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Instagram
          required:
            - Comment
          properties:
            Comment:
              type: string
              description: Please input a comment you have
              default: stringNew
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    tempComment = new_task['Comment']

    X_New = np.array([tempComment])

    clf = joblib.load('static/InstagramComment.pkl')

    resultPredict = clf[0].predict(X_New)
    # target_names = ['yes', 'no']
    # return jsonify({'message': format(clf[1].Label[resultPredict])})
    return jsonify({'message': format(resultPredict[0])})

app.run()