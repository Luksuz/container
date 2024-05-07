import torch
from PIL import ImageDraw

class RegionDetectionModel:
    weights_path = "weights/last.pt"
    
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=RegionDetectionModel.weights_path)
        
    def _get_predictions(self, img, conf=0.4):
        self.model.conf = conf
        with torch.no_grad():
            preds = self.model(img)
        labels = preds.xyxy[0]
        return labels, img
    
    def get_serial_region(self, img,  show=False):
        labels, img = self._get_predictions(img)
        if len(labels) == 0:
            return False
        
        best_label = max(labels, key=lambda x: x[5])

        x1 = best_label[0].item(); y1 = best_label[1].item()
        x2 = best_label[2].item(); y2 = best_label[3].item()
        
        detected = img.crop((x1, y1, x2, y2))
        filepath = "detected.jpg"
        detected.save(filepath)
        
        if show:
            for label in labels:
                x1 = label[0].item(); y1 = label[1].item()
                x2 = label[2].item(); y2 = label[3].item()

                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            img.show()
            
        return filepath
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