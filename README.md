# MNIST Handwritten Digit Classification using Keras Sequential API

This project builds and trains a deep learning model using the **Keras Sequential API** to classify handwritten digits (0–9) from the **MNIST dataset**.  
Along with building a neural network, the project performs **extensive data augmentation**, visualization, evaluation, and model analysis to understand model behavior deeply.

---

## Project Overview

- Loaded MNIST dataset using TensorFlow/Keras  
- Normalized image pixel values  
- Performed **seven powerful data augmentations** to increase dataset variability  
- Built a deep neural network using **Dense layers + Batch Normalization**  
- Trained for 30 epochs with validation monitoring  
- Evaluated performance using multiple metrics  
- Visualized model confidence, ROC curves, PR curves, and misclassifications  
- Saved the final trained model using Joblib  

---

## Dataset  
- **Dataset:** MNIST (70,000 grayscale 28×28 handwritten digit images)  
- **Classes:** 10 (Digits 0–9)  
- **Train/Test Split:** 80% Training, 20% Testing  
- All pixel values were scaled to the range **0–1**

---

## Exploratory Data Analysis

The following plots were generated:

- Single and grid visualization of handwritten digits  
- Training sample distribution  
- Test sample distribution  

These helped in understanding the dataset before training the model.

---

## Data Augmentation Techniques

Seven augmentations were applied to improve model generalization:

1. **Shift Left (2 px)**  
2. **Shift Right (2 px)**  
3. **Shift Up (2 px)**  
4. **Shift Down (2 px)**  
5. **Zoom In**
6. **Zoom Out**
7. **Original Images**

The augmented training dataset size became **7× larger**, giving the model robust exposure to real-world handwriting variations.

---

## Model Architecture (Keras Sequential)

The neural network consists of:

- Input layer: 784 neurons  
- Dense(300) + BatchNorm  
- Dense(200) + BatchNorm  
- Dense(100) + BatchNorm  
- Dense(50) + BatchNorm  
- Output: Dense(10, softmax)

This architecture combines **depth + normalization**, enabling stable and high-accuracy learning.

---

## Training History

Plotted:

- Training loss vs. validation loss  
- Training accuracy vs. validation accuracy  

These curves helped detect overfitting or underfitting patterns.

---

## Model Evaluation Metrics

| Metric                   | Value    |
|--------------------------|----------|
| **Test Accuracy**        | *98.84%* |
| **Precision (Weighted)** | *98.84** |
| **Recall (Weighted)**    | *98.84** |
| **F1 Score (Weighted)**  | *98.84** |
| **ROC-AUC Score (OvR)**  | *0.9996* |
| **Log Loss**             | *0.085*  |
| **Cohen’s Kappa Score**  | *98.71*  |
| **Matthews Corr.Coeff.** | *98.71*  |
| **Top-3 Accuracy**       | *99.88%* |

---

## Visualizations Produced

- Confusion Matrix  
- Per-Class Accuracy  
- ROC Curve (per digit)  
- Precision-Recall Curve (per digit)  
- Calibration curves for all 10 classes  
- Histogram of prediction confidence  
- First 25 correctly classified samples (>= 90% confidence)  
- First 25 misclassified samples  

These visualizations provide deep insight into both strengths and weaknesses of the model.

---

## Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras (Sequential API)  
- **Data Manipulation:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Model Persistence:** Joblib  
- **Utilities:** SciPy (for zoom-based augmentation)
- **Scikit-learn** (metrics, preprocessing, splitting)

## Saving the Model

The final trained Keras model is saved using Joblib:

`python
import joblib
joblib.dump(model, 'Sequential_Perceptron_model.pkl')`

---
## Author  

**Lavan Kumar Konda**  
-  2nd Year Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)

