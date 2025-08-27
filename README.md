# Dental-Caries-Detection-Using-Deep-Learning
A deep learning system for automated dental caries detection in bitewing X-rays using YOLOv8. This project achieved 90.1% accuracy, providing a tool to assist in early and efficient diagnosis. Developed as a B.Tech final year project.
# Dental Caries Detection Using Deep Learning

This repository contains the code and resources for a Bachelor of Technology final year project conducted at the National Institute of Technology Manipur. The project focuses on leveraging deep learning, specifically the YOLOv8 architecture, to automatically detect and localize dental caries in bitewing radiographs.

## 📖 Abstract

Dental caries is a widespread oral health issue. Traditional diagnostic methods are often subjective and time-consuming. This project explores the application of deep learning to enhance the accuracy and efficiency of caries detection. We developed a model based on YOLOv8, trained on a curated dataset of dental X-ray images, to automatically identify and localize various lesions. The model demonstrated high performance, suggesting its potential as a valuable tool for assisting dental professionals in diagnostics.

## ✨ Features

- **State-of-the-art Model:** Utilizes Ultralytics YOLOv8 for object detection and segmentation.
- **Comprehensive Preprocessing:** Implements auto-orientation, resizing, and extensive image augmentation.
- **Robust Performance:** Achieved an accuracy of **90.1%**, with high precision and recall rates.
- **Clinical Relevance:** Aims to provide a non-invasive, efficient, and reliable method for early caries detection.

## 📊 Results

The trained model achieved the following performance metrics on the test set:

| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Dental Caries  | 97.3%     | 85.7%  | 91.1%    |
| No Caries      | 82.4%     | 96.6%  | 88.9%    |
| **Accuracy**   |           |        | **90.1%**|

**Confusion Matrix:**
![Confusion Matrix](path/to/confusion_matrix.png) <!-- Upload your image and update the path -->

## 🗂️ Dataset

The model was trained on a dataset of **2890 bitewing radiographs**, sourced from platforms like Kaggle and Mendeley Data.

**Dataset Split:**
- **Training Set:** 2592 images (90%)
- **Validation Set:** 177 images (6%)
- **Test Set:** 121 images (4%)

All images were preprocessed and resized to **640x640 pixels**.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8 or later
- pip

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dental-caries-detection.git
   cd dental-caries-detection
2. Install required dependencies:
   pip install -r requirements.txt

   Example requirements.txt:
     ultralytics
     opencv-python
     matplotlib
     seaborn
     numpy
     torch
     torchvision
3.Run inference on a new image:

  python predict.py --weights best_model.pt --source path/to/your/image.jpg
  (Ensure you have your trained model weights best_model.pt in the project directory)




📁 Project Structure

dental-caries-detection/
├── data/
│   ├── train/                 # Training images and labels
│   ├── valid/                 # Validation images and labels
│   └── test/                  # Test images
├── notebooks/                 # Jupyter notebooks for EDA and training
├── src/
│   ├── preprocess.py          # Image preprocessing and augmentation
│   ├── train.py               # Script to train the YOLOv8 model
│   └── predict.py             # Script to run inference on new images
├── runs/
│   └── detect/
│       └── train/             # Output of training (weights, plots, results)
├── requirements.txt
└── README.md

🧪 Methodology
1.Data Preprocessing: Auto-orientation based on EXIF data and resizing to 640x640 px.

2.Data Augmentation: Applied flipping, rotation, cropping, shear, grayscale conversion, and noise addition to improve model generalization.

3.Model: Fine-tuned a pre-trained YOLOv8 model on our dental caries dataset.

4.Training: The model was trained using a GPU, leveraging the Ultralytics framework.

5.Evaluation: Performance was assessed using standard metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.



👥 Contributors
1.Catherine Nembiakching (20103057)

2.Daniel Lunginlian (20103046)

3.Ramavath Vinod Naik (20103045)

Supervisor: Mr. Sanabam Bineshwor Singh, Lecturer, CSE Department, NIT Manipur.



📜 License
This project is licensed under the MIT License - see the LICENSE file for details. All rights reserved (2024).

.

🔗 Citation
If you use this project in your research, please cite the accompanying report:

C. Nembiakching, D. Lunginlian, R. V. Naik, "Dental Caries Detection Using Deep Learning",
B.Tech Thesis, National Institute of Technology Manipur, 2024.


📚 References
Key references from the project report are included in the References section of the report. This work builds upon previous research in deep learning for medical image analysis, particularly in dentistry.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Disclaimer: This model is intended for research purposes and should not be used as a sole diagnostic tool in a clinical setting without further validation by medical professionals.









