# 🔥 Kosmos-2 Fire/Smoke Detection App

This project contains:
- A Streamlit app to upload RGB images and generate grounded captions using Kosmos-2
- A fine-tuning script to retrain the model on custom fire/smoke datasets
- Docker support to run everything as a containerized service

## 📁 Folder Structure

- `app/` – Streamlit app and Dockerfile
- `fine_tune/` – Training pipeline
- `data/` – Input images and captions
- `outputs/` – Model checkpoints

## ▶️ Run Streamlit App

```bash
cd app
streamlit run app.py
