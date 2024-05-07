import torch
from PIL import Image, ImageDraw

class CharactersDetectionModelV2:
    weights_path = "weights/best_characters.pt"
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
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=CharactersDetectionModelV2.weights_path)
        
    def _get_predictions(self, img, conf=0.4):
        self.model.conf = conf
        img = Image.open(img)
        with torch.no_grad():
            preds = self.model(img)
        return preds, img
    
    def sort_and_read_characters(self, img, conf=0.4, show=False):
        preds, img = self._get_predictions(img, conf)
        sorted_labels = sorted(sorted(preds.xyxy[0], key=lambda x: x[4])[:11], key=lambda x: x[0])
        for label in sorted_labels:
            print(label)
        
        if show:
            for label in sorted_labels:
                x1 = label[0].item(); y1 = label[1].item()
                x2 = label[2].item(); y2 = label[3].item()

                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            img.show()
            
        chars = [str(self.model.names[int(x[-1])]) for x in sorted_labels[:11]]
        print(chars)

            
        for i in range(len(chars[:4])):
            chars[i] = CharactersDetectionModelV2.digit_to_char.get(chars[i], chars[i])
        for i in range(4, len(chars[4:11])):
            chars[i] = CharactersDetectionModelV2.char_to_digit.get(chars[i], chars[i])
        chars = f"{''.join(chars[:4])}-{''.join(chars[4:-1])}-{chars[-1]}"
        return chars
"""

img1_path = Path("/Users/luksuz/Desktop/container_ocr/BBCU5100638.jpeg")
img2_path = Path("/Users/luksuz/Desktop/container_ocr/NYKU4576570.jpg")
img3_path = Path("/Users/luksuz/Desktop/container_ocr/ICOU6024116.png")
img4_path = Path("/Users/luksuz/Desktop/container_ocr/LSGU1077379.jpeg")
img5_path = Path("/Users/luksuz/Desktop/container_ocr/DSCF1594.jpg")
img = Image.open(img5_path)
model = RegionDetectionModel()
model.get_serial_region(img)
        """