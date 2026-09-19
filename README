🧠 Brain Tumor MRI Classification with Transfer Learning

A deep learning project that classifies brain MRI scans into four categories — glioma, meningioma, pituitary tumor, or no tumor — using transfer learning with three pretrained CNN architectures: VGG19, InceptionV3, and ResNet152. Includes a comparison of model performance and a Streamlit web app for interactive predictions.

📊 Results

All three models were fine-tuned on the dataset and evaluated on a held-out test set, each exceeding 80% test accuracy:

Model	Test Accuracy	Test Loss
VGG19	88.75%	0.7116
ResNet152	86.56%	0.6679
InceptionV3	84.50%	0.4723

Show Image

🗂️ Dataset

Brain Tumor MRI Dataset (Kaggle) — ~7,000 MRI images across 4 classes:

glioma
meningioma
notumor
pituitary

Downloaded automatically via kagglehub in the notebook.

🧪 Methodology
Preprocessing — images resized to 150×150, split into train/validation (80/20) and a separate held-out test set, with tf.data caching and prefetching for performance.
Augmentation — random horizontal flip, rotation, zoom, and contrast applied during training to reduce overfitting.
Transfer learning — each backbone (VGG19 / InceptionV3 / ResNet152) loaded with ImageNet weights, base frozen, and a custom classification head added (Global Average Pooling → Dense(256, ReLU) → Dropout(0.3) → Dense(4, Softmax)).
Two-phase training:
Phase 1: train only the classification head (base frozen).
Phase 2: unfreeze the top ~20% of the base model's layers and fine-tune at a lower learning rate.
Evaluation — final accuracy/loss compared across train, validation, and test sets for all three models.
📁 Repository Structure
.
├── brain_tumor_classification.ipynb   # Training notebook (data, models, training, comparison)
├── app.py                             # Streamlit app for interactive predictions
├── requirements.txt                   # Python dependencies
├── models/                            # (not included) saved .keras model files
└── README.md
⚙️ Setup
bash
git clone https://github.com/<your-username>/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt
1. Train the models

Open and run brain_tumor_classification.ipynb (Google Colab or Jupyter with a GPU is recommended). At the end of training, save each model so the app can load it:

python
model.save("models/vgg19_brain_tumor.keras")
model.save("models/resnet152_brain_tumor.keras")
model.save("models/inceptionv3_brain_tumor.keras")
2. Run the Streamlit app

Make sure the saved .keras files are in a models/ folder next to app.py, then run:

bash
streamlit run app.py

Upload an MRI image, pick a model, and get a prediction with a class probability breakdown.

⚠️ Disclaimer

This project is for educational purposes only. It is not a validated medical device and must not be used for real diagnostic or clinical decisions.

🚀 Possible Improvements
Confusion matrix and per-class precision/recall/F1
Grad-CAM visualizations for model interpretability
Hyperparameter tuning / cross-validation
Ensemble of all three models
📄 License

This project is released under the MIT License.
