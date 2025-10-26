# 👁️ Multiclass Ocular Disease Prediction using Transfer Learning Models

## 📘 Overview
This project presents a **deep learning–based multiclass ocular disease prediction system** using transfer learning models.  
The aim is to detect and classify multiple eye diseases such as **Cataract**, **Glaucoma**, and **Diabetic Retinopathy** from retinal fundus images, enabling early diagnosis and reducing vision loss risk.

The project leverages state-of-the-art convolutional neural networks (CNNs) — **MobileNetV2**, **Efficient CNN**, and **VGG19** — with an **attention mechanism** to enhance classification accuracy.  
The best-performing model, **VGG19 with Attention**, achieved an overall accuracy of **93%**, making it suitable for potential clinical diagnostic applications.

---

## 🎯 Objectives
- Enable **early detection** of ocular diseases using automated image analysis.  
- Classify multiple eye conditions from retinal images with high accuracy.  
- Compare and evaluate **different CNN architectures** for optimal performance.  
- Design a **computationally efficient** and scalable diagnostic model accessible even in limited-resource settings.  

---

## 📂 Dataset
Dataset used: [Eye Diseases Classification - Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

### Classes:
- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**

Each folder contains retinal fundus images for that specific class.  
The dataset includes ~4,200 images distributed evenly across categories.

---

## 🧠 Methodology
### 🧾 Workflow
1. **Data Collection and Preprocessing**
   - Collected images from the Kaggle dataset.
   - Resized all images to **224×224** pixels.
   - Normalized and augmented data (rotation, zoom, flips) to improve diversity.
   - Split dataset: **80% training**, **20% testing**.

2. **Model Training**
   - Implemented and compared **three CNN architectures**:
     - MobileNetV2
     - Efficient CNN (custom)
     - VGG19
   - Applied **transfer learning** with pretrained ImageNet weights.
   - Used **categorical cross-entropy loss** and **Adam optimizer**.

3. **Attention Mechanism**
   - Added an **attention layer** before the final dense layers in VGG19.
   - Enhanced feature extraction by focusing on relevant image regions.
   - Improved classification performance and reduced loss.

4. **Evaluation**
   - Evaluated accuracy, loss, precision, recall, and F1-score.
   - Compared confusion matrices and learning curves.

---

## 🧩 Model Architectures

### 🔹 MobileNetV2
- Lightweight pretrained CNN with fine-tuning for 4-class classification.
- Achieved **82% accuracy**.

### 🔹 Efficient CNN (Custom)
- Three convolutional layers with ReLU activations and max pooling.
- Flatten + Dense layers for classification.
- Achieved **86% accuracy**.

### 🔹 VGG19
- Deep CNN architecture with additional Dense and BatchNorm layers.
- Trained with and without attention.
- Without attention: **91.6% accuracy**  
- With attention: **93% accuracy (best model)**

---

## ⚙️ Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | OpenCV, ImageDataGenerator |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 📊 Results

| Model | Accuracy | Key Observations |
|:------|:----------|:----------------|
| MobileNetV2 | 82% | Struggled with similar features between diseases |
| Efficient CNN | 86% | Better generalization, but less feature depth |
| VGG19 | 91.6% | Strong baseline performance |
| **VGG19 + Attention** | **93%** | Focused on critical regions and achieved best accuracy |

### Visualization Outputs:
- **Training vs Validation Accuracy and Loss** curves  
- **Confusion Matrices** for each architecture  
- **Predicted vs True Labels**  
- **Sample Prediction Results**  

Example output:  
Predicted: Glaucoma  
Actual: Glaucoma  
Confidence: 98.2%  


---

## 🚀 How to Run

### 🔧 Prerequisites
Install required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python pillow
```

## ▶️ Steps

### 🪜 Clone this repository
### 📦 Download and extract the dataset from Kaggle

🔗 [Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

## 📈 Performance Summary

| Model | Accuracy | Loss | Epochs |
|:------|:----------|:-----|:-------|
| MobileNetV2 | 82% | 0.35 | 15 |
| Efficient CNN | 86% | 0.29 | 15 |
| VGG19 | 91.6% | 0.24 | 20 |
| **VGG19 + Attention** | **93%** | **0.22** | 20 |

---

### 🧮 Evaluation Metrics
- **Precision:** 0.93  
- **Recall:** 0.92  
- **F1-Score:** 0.93  

---

## 🔍 Insights
- **VGG19 with attention mechanism** showed superior feature localization.  
- **Data augmentation** improved generalization and prevented overfitting.  
- **Attention layers** effectively emphasized critical fundus regions related to diseases.  
- The model can be integrated into an **early diagnostic screening tool** for ophthalmology.  

---

## 🧭 Future Enhancements
- ⚡ Implement **transfer learning** with advanced architectures (ResNet50, InceptionV3).  
- 🌐 Develop a **web interface** for real-time image uploads and predictions.  
- 🩺 Extend dataset to include more disease types (e.g., AMD, Myopia).  
- 🔬 Incorporate **Grad-CAM** visualizations for explainable AI insights.  


