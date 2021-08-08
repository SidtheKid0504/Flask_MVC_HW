from flask import Flask, request, jsonify
from model import predict_input_image

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_data():
    image = request.files.get('alphabet')
    prediction = predict_input_image(image)
    try:
        return jsonify({
            "prediction": prediction
        }), 200
    except:
        return ""

if __name__ == "__main__":
    app.run(debug=True)