# Container Serial Number Detection Server

## Overview

Welcome to the Container Serial Number Detection server repository! This project uses two YOLOv5 models for which i personally collected and labeled the dataset to detect and recognize serial numbers on shipping containers. The first model identifies the serial number region, and the second model takes that region as input to detect and classify the characters with a 96% mean average precision.

## Features

- **Serial Number Region Detection**: Identifies the region containing the serial number on a shipping container.
- **Character Detection and Classification**: Recognizes and classifies the characters within the detected serial number region.
- **High Precision**: Achieves a 96% mean average precision in character detection and classification.
- **RESTful API**: Provides an API for uploading container images and retrieving detected serial numbers.

## Getting Started

To run the Container Serial Number Detection server locally, follow these steps:

1. **Clone the repository**:

```bash
cd app
pip install -r requirements.txt
python main.py
```

API Endpoints

	•	POST /container: Upload an image of a shipping container.
	•	Request: Multipart/form-data with an image file.
	•	Response: JSON with the detected serial number, inference time and the image of the serial number region.

Technologies Used

	•	Backend: Flask for the web server
	•	Machine Learning: YOLOv5 for object detection
	•	Python Libraries: OpenCV, PyTorch
	•	Deployment: (Optional: Describe how the server is deployed, e.g., Docker, AWS EC2)

Contributing

Contributions to enhance or fix issues in the Container Serial Number Detection server are welcome. If you have suggestions, improvements, or bug fixes, please feel free to submit a pull request or raise an issue.
