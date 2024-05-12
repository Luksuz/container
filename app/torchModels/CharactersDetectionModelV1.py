from PIL import Image, ImageDraw
from roboflow import Roboflow

class CharactersDetectionModelV1:
    digit_to_char = {
        "1": "I",
        "2": "Z",
        "3": "B",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "O"
    }
    char_to_digit = {
        "A": "4",
        "B": "8",
        "C": "0",
        "D": "0",  # This could be interpreted as "O"
        "E": "3",
        "F": "3",  # This could be interpreted as "E"
        "G": "6",
        "H": "0",  # This could be interpreted as "I"
        "I": "1",
        "J": "1",  # This could be interpreted as "I"
        "K": "0",  # This could be interpreted as "X"
        "L": "1",  # This could be interpreted as "I"
        "M": "0",  # This could be interpreted as "W"
        "N": "7",  # This could be interpreted as "U"
        "O": "0",
        "P": "9",  # This could be interpreted as "D"
        "Q": "0",  # This could be interpreted as "O"
        "R": "0",  # This could be interpreted as "P"
        "S": "5",
        "T": "1",  # This could be interpreted as "I"
        "U": "0",
        "V": "0",  # This could be interpreted as "U"
        "W": "0",
        "X": "0",
        "Y": "1",  # This could be interpreted as "V"
        "Z": "2"
        # Add more homoglyphs as needed
    }
    
    def __init__(self):
        self.model = self._load_model()
        
    def _load_model(self):
        rf = Roboflow(api_key="lvLBDUDGR6d8W7tdX6Dq")
        project = rf.workspace().project("container-characters-detection")
        model = project.version(4).model
        return model
        
    def _get_predictions(self, img_path, conf):
        print(img_path)
        img = Image.open(img_path)
        labels = self.model.predict(img_path, confidence=conf, overlap=30).json()["predictions"]
        return labels, img
    
    def draw_bbs(self, labels, img):        
        draw = ImageDraw.Draw(img)
        for label in labels:
            x, y = label["x"], label["y"]
            w, h = label["width"]/2, label["height"]/2
            x1, y1, x2, y2 = x-w, y-h, x+w, y+h
            
            draw.rectangle((x1,y1,x2,y2), outline=(0,255,0), width=1)
            
        self.img.show()
        
    def sort_and_read_characters(self, img_path, conf=40, show=False):
        labels, img = self._get_predictions(img_path, conf)
        shape = img.size
        
        if shape[0] > shape[1]:  # Height is greater than width
            sorted_preds = sorted(labels, key=lambda x: x["x"])
        else:
            sorted_preds = sorted(labels, key=lambda x: x["y"])
        chars = [str(pred["class"]) for pred in sorted_preds[:11]]

        for i in range(len(chars[:4])):
            chars[i] = CharactersDetectionModelV1.digit_to_char.get(chars[i], chars[i])
        for i in range(4, len(chars[4:11])):
            chars[i] = CharactersDetectionModelV1.char_to_digit.get(chars[i], chars[i])
        chars = f"{''.join(chars[:4])}-{''.join(chars[4:-1])}-{chars[-1]}"

        if show:
            self.draw_bbs(labels, img)

        return chars  
    
"""
img1_path = "detected.jpg"
model = CharactersDetectionModel()
result = model.sort_and_read_characters(img1_path)
print(result)
"""
        