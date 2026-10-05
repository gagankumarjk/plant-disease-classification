# 🌿 Plant Leaf Disease Detection

Streamlit application for plant leaf disease classification using the fine-tuned MobileNetV2 model from Capstone Project 2.

## Model
- Architecture: MobileNetV2
- Transfer learning: ImageNet pretrained weights
- Fine-tuning: final 40 layers
- Input: 224 × 224 RGB
- Classes: 23
- Balanced working dataset: 3,450 images (150 per class)
- Held-out test accuracy: 89.19%

## Files
- `app.py` — Streamlit application
- `best_plant_disease_224_finetuned.keras` — trained model
- `plant_disease_class_names.json` — 23 class labels
- `requirements.txt` — Python dependencies

## Run locally with Anaconda Prompt

```bash
conda create -n plant_disease_app python=3.11 -y
conda activate plant_disease_app
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## GitHub

```bash
git init
git add .
git commit -m "Initial plant disease Streamlit application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-classification.git
git push -u origin main
```

## Application workflow

Upload or capture a leaf image → OpenCV decoding and RGB conversion → resize to 224 × 224 → MobileNetV2 prediction → disease and confidence → Top-3 predictions.

## Disclaimer

This application is an educational/project demonstration. Performance on real-world field images may differ from the PlantVillage test result because of background, lighting, camera quality and domain shift.
