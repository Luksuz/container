from flask import Flask, request
#from flask_cors import CORS
from PIL import Image
import time

from torchModels.RegionDetectionModel import RegionDetectionModel
from torchModels.CharactersDetectionModelV1 import CharactersDetectionModelV1
import responseDTO

app = Flask(__name__)
#CORS(app)

regionDetectionModel = RegionDetectionModel()
charactersDetectionModel = CharactersDetectionModelV1()  # CharactersDetectionModelV2

@app.route("/predict/v1", methods=["POST"])
def predict_v1():
    start = time.time()
    image = request.files.get('image')
    
    if not image:
        raise ValueError("No image file provided.")

    if not image.filename.endswith(("jpg", "png", "jpeg", "webp")):
        raise ValueError("Invalid file type. Supported types: jpg, png, jpeg, webp.")

    image = Image.open(image)
    image = image.convert("RGB")
    filepath, images = regionDetectionModel.get_serial_region(image)

    if not filepath:
        return responseDTO.NO_PREDICTIONS

    chars = charactersDetectionModel.sort_and_read_characters(filepath, show=False)
    end = time.time()

    speed = end - start
    return responseDTO.get_predictions(chars, images, speed)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
