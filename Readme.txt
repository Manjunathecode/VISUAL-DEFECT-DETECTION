# 🔍 AI-Powered Visual Defect Detection System

This project is a deep learning-based computer vision system designed to automatically detect defects in industrial casting products.

## 🚀 Overview

Manual inspection in manufacturing is time-consuming and prone to human error. This system uses Artificial Intelligence to automate defect detection with high accuracy.

The model classifies product images into:

* ❌ Defective
* ✅ Non-Defective

## 🧠 Model Details

* Architecture: ResNet18 (Transfer Learning)
* Framework: PyTorch
* Dataset: Casting Product Image Dataset (1300 images)
* Training Accuracy: 99.04%
* Test Accuracy: 98.85%

## ⚙️ Features

* Image classification using CNN
* Transfer learning for better performance
* Real-time prediction using Gradio web interface
* Confidence score output
* Easy-to-use UI for uploading images

## 🏗️ Tech Stack

* Python
* PyTorch
* Torchvision
* Gradio
* NumPy, Matplotlib, Scikit-learn

## 🔄 Workflow

1. Load dataset
2. Preprocess images (resize, normalize)
3. Train ResNet18 model
4. Evaluate model performance
5. Save trained model
6. Deploy using Gradio interface

## 💻 How to Run

1. Clone the repository
2. Install dependencies
3. Run the app:

```bash
python app.py
```

4. Upload an image and get prediction

## 📊 Output

The system predicts whether a product is defective or not along with a confidence score.

## 📌 Future Improvements

* Defect localization (Object Detection)
* Real-time camera integration
* Multi-class defect classification
* Deployment on edge devices

## 📄 Project Report

Detailed report available in the repository: 

---

**Author:** Manjunath S Vernekar
