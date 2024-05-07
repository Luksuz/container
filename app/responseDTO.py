INVALID_FILE_TYPE = {"code": 400,
                "message": "Invalid file type! Make sure that the file is an image. Supported types include: JPG, JPEG, WEBP, PNG"}  

NO_PREDICTIONS = {"code": 400,
                "message": "The model was unable to make a prediction"}  

def get_predictions(chars, speed):
    return {"serial_number":chars,
            "length": len(chars.replace("-", "")),
            "speed": f"{speed:.2f} seconds.",
            "code": 200}  