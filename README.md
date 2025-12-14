# Image Text VQA Model - Multimodal Mini Project

A lightweight Visual Question Answering (VQA) system that combines computer vision and natural language processing to answer questions based on input images. This project integrates **CLIP** for image feature extraction and a **small text model (DistilGPT2)** for text generation.

## 📌 Project Overview

This project implements a generative VQA architecture designed for efficiency and semantic accuracy. It was built as a self-directed mini-project to explore multimodal learning and optimization techniques.

### Key Features
* **Multimodal Architecture**: Fuses **OpenAI CLIP** (Vision Transformer) embeddings with **DistilGPT2**.
* **Mixed Precision Training**: Implements `torch.amp` (Automatic Mixed Precision) to optimize memory usage and speed up training by **2x** on compatible hardware.
* **Robust Evaluation**: Uses **BERTScore** to evaluate the semantic similarity between generated answers and ground truth, providing a better metric than standard accuracy or BLEU scores.
* **Hyperparameter Tuning**: optimized for performance on consumer-grade GPUs/CPUs.

## 📂 Project Structure

```text
├── config.py           # Central configuration for hyperparameters and paths
├── dataset.py          # Custom PyTorch Dataset class for loading images and text
├── model.py            # Neural network architecture (CLIP + Projection + GPT2)
├── train.py            # Training loop with Mixed Precision support
├── evaluate.py         # Inference script using BERTScore for metrics
├── utils.py            # Helper functions for text cleaning and formatting
├── requirements.txt    # List of dependencies
└── data/               # Directory for dataset storage
    ├── images/         # Folder containing image files
    └── vqa_dataset.csv # CSV file with image_id, questions, and answers
