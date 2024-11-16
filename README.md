**A DeepLearning Approach for Classifying Diabetic Retinopathy Stages**

![image](https://github.com/user-attachments/assets/019bf5bc-7c23-4f38-8753-993d3017cfe3)


Diabetic Retinopathy (DR) is a severe complication of diabetes that can lead to permanent blindness if left untreated. This project leverages deep learning techniques to automate the detection of DR stages from retinal fundus images, providing a scalable solution to assist healthcare professionals in early diagnosis and prevention of vision loss.

🌟 **Key Features**

Multi-Class Classification: Detection of five DR stages: No DR, Mild, Moderate, Severe, and Proliferative DR.
Advanced CNN Models: Utilized EfficientNet and Multi-Level EfficientNet for optimal performance.
Data Augmentation: Real-time augmentations including rotation, flipping, and brightness adjustments to enhance generalization.
Ensemble Learning: Combined predictions from multiple models to improve accuracy and stability.
Comprehensive Evaluation: Assessed using sensitivity, specificity, and AUROC.

**📂 Dataset**

The dataset for this study is the publicly available APTOS 2019 Blindness Detection dataset, containing:
3,662 high-resolution images of retinal fundus.
Five Class Labels:
0: No DR
1: Mild DR
2: Moderate DR
3: Severe DR
4: Proliferative DR
Dataset Challenges:
Imbalanced Distribution: Majority of images labeled as "No DR," with fewer samples in advanced stages.

**🧠 Network Architecture**

*Model Design:*
EfficientNet:

Pre-trained on ImageNet for feature-rich representations.
Fine-tuned for medical imaging tasks.

*Multi-Level EfficientNet:*

![Screenshot 2024-11-11 163504](https://github.com/user-attachments/assets/66ccd463-1b1d-472e-9f61-10e8d8064c11)

Captures layered features to distinguish subtle differences across DR stages.

*Ensemble Model:*
Combines predictions from both models via averaging to improve reliability and reduce variance.

*Model Enhancements:*
Dropout Layers to prevent overfitting.
Fine-tuning to adapt pre-trained weights to the unique patterns in the dataset.

**🔄 Data Preprocessing**
Steps:
Resizing: All images resized to 400x400 pixels for uniformity.
Gaussian Blur: Applied to reduce noise and enhance key features.

**🔄 Data Augmentation**
Dynamic augmentations were applied to each image during training to improve generalization:
Random rotation
Horizontal and vertical flipping
Brightness adjustments

**⚙️ Training Process**
Initialization: Models initialized with ImageNet pre-trained weights.
Optimizer: Used Adam optimizer with an initial learning rate of 
Learning Rate Scheduler: Dynamically adjusted learning rates for effective convergence.
Early Stopping: Halted training based on validation performance to avoid overfitting.
Loss Function: Cross-entropy loss for multi-class classification.
Hardware: Training executed on Kaggle GPU (P100) for efficiency.

**📊 Evaluation Metrics**
The models were evaluated using:
Sensitivity
Specificity
Area Under the Receiver Operating Characteristic (AUROC) Curve

**🚀 Results**
The ensemble model demonstrated:
Improved classification accuracy across all stages of DR.
Reduced variance in predictions for more stable performance.

**📂 Repository Structure**
``` bash
├── notebooks/                # Jupyter notebooks for EDA and experimentation  
├── README.md                 # Project documentation 
```
**Tools & Libraries**

tensorflow==2.12.0

numpy==1.23.1

pandas==1.5.2

scikit-learn==1.0.2

matplotlib==3.5.0


**🔗 References**

APTOS 2019 Blindness Detection Dataset: Kaggle Challenge [APTOS DATASET](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

EfficientNet: Paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

Reproducing the Paper[Deep Learning-Based Detection of Referable Diabetic Retinopathy and Macular Edema Using Ultra-Widefield Fundus Imaging](https://arxiv.org/abs/2409.12854)

**🤝 Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

**📧 Contact**
For any queries, feel free to reach out at sunilkumar.a@ihub-data.iiit.ac.in
