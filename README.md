# Acne Detection on Skin using YOLOv5

## Project Overview
This project aims to assist users in detecting acne and skin irregularities using the YOLOv5 object detection model. It provides two options for analysis:

- **Image Upload:** Users can upload a static image for acne detection.
- **Live Video Feed:** Real-time detection through a live video stream.

This tool serves as a preliminary assessment for users before consulting a dermatologist.

## Models
The project utilizes the YOLOv5 model, known for its efficiency and accuracy in object detection tasks. The model is trained to detect acne patterns and irregularities on the skin.

## How it Works
1. Users can either upload an image or enable their webcam for live video analysis.
2. The YOLOv5 model processes the input to detect acne and skin irregularities.
3. Results are displayed in real-time or after image processing.

## Steps to Execute
1. Download the entire project zip file or clone the repository.
   ```bash
   git clone <repository-link>
   ```
2. Download the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook file to initialize and test the YOLOv5 model:
   ```bash
   jupyter notebook yolov511.ipynb
   ```
4. After completing the Jupyter Notebook execution, run the Python script for real-time testing:
   ```bash
   python yolov511.py
   ```
5. Test the model using either image uploads or live video feed.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- OpenCV
- YOLOv5 Dependencies
- NumPy
- Matplotlib

You can install them using:
```bash
pip install -r requirements.txt
```

## Future Improvements
- Enhance model accuracy with a larger dataset.
- Add additional skin condition detection capabilities.
- Develop a mobile-friendly version of the application.

## Acknowledgments
Special thanks to the open-source YOLOv5 community for providing pre-trained models and resources.

## **Screenshots**
![image](https://github.com/user-attachments/assets/4a9fd933-0318-4a76-adf2-08924586196b)

![image](https://github.com/user-attachments/assets/16477f3a-df1d-40a2-82c4-843da81f388c)



---
For any issues or contributions, feel free to open an issue or submit a pull request.

