# 🏏 Cricket Commentary Generator

An innovative deep learning system that automatically generates dynamic and professional cricket commentary for video clips. Leveraging **CLIP** for visual understanding and **DeepSeek** for natural language generation, this project integrates advanced features like **temporal encoding**, **gradient accumulation**, and **mixed precision training** to deliver high-quality, real-time commentary, enhanced with realistic **crowd ambiance**.

---

## 📋 Table of Contents

1.  [About the Project](#about-the-project)
2.  [Features](#features)
3.  [Architectural Overview](#architectural-overview)
4.  [Project Structure](#project-structure)
5.  [Key Components Explained](#key-components-explained)
6.  [Training Pipeline](#training-pipeline)
7.  [Inference Pipeline](#inference-pipeline)
8.  [Setup and Installation](#setup-and-installation)
9.  [Usage](#usage)
10. [Results & Demos](#results--demos)
11. [Model Weights](#model-weights)
12. [License](#license)
13. [Contributing](#contributing)

---

## 🌟 About the Project

The Cricket Commentary Generator aims to automate the creation of engaging cricket commentary from video footage. By integrating computer vision with large language models, the system analyzes video sequences of cricket deliveries, understands the action, and then generates concise, professional, and contextually relevant commentary. The output is further enhanced by mixing it with stadium ambiance, creating an immersive audio-visual experience.

This project addresses the challenges of real-time sports broadcasting and content creation by providing an efficient, AI-driven solution for generating commentary, reducing manual effort and speeding up content delivery.

---

## ✨ Features

* **Video Understanding:** Utilizes **CLIP (Contrastive Language–Image Pre-training)** to extract meaningful visual features from cricket video frames.
* **Temporal Encoding:** Employs a **Temporal Transformer Encoder** to capture the sequential nature and evolution of events within the video, providing a comprehensive understanding of the cricket action.
* **Advanced Language Generation:** Integrates **DeepSeek-Coder-1.3B-Instruct** as the language model to generate professional and contextually accurate cricket commentary.
* **Optimized Training:** Implements **gradient accumulation**, **mixed precision training (AMP)**, and a **linear learning rate scheduler with warm-up** for stable and efficient model training.
* **Robust Fine-tuning:** Selectively unfreezes and fine-tunes the last layers of the DeepSeek model for specialized commentary generation.
* **Intelligent Commentary Summarization:** Leverages the **Groq API (Llama 3.1 8B Instant)** to refine and summarize the generated commentary into concise, broadcast-ready sentences.
* **Text-to-Speech Integration:** Uses **ElevenLabs API** to convert the summarized commentary into natural-sounding speech.
* **Audio Mixing:** Seamlessly blends the AI-generated commentary with **stadium ambiance sounds** to create a realistic broadcast feel.
* **User-Friendly Interface:** Provides a simple **Gradio** interface for easy video uploads and commentary generation.

---

## 🧠 Architectural Overview

The system operates through a sophisticated multi-modal architecture:

1.  **Video Frame Extraction & Preprocessing:** Video segments are processed, and relevant frames are extracted and preprocessed using CLIP's `preprocess` pipeline.
2.  **Visual Feature Extraction (CLIP):** The preprocessed frames are fed into a frozen **CLIP ViT-B/32** model to obtain powerful visual embeddings.
3.  **Temporal Feature Aggregation (Temporal Transformer Encoder):** These frame-level CLIP features are then passed through a custom **Temporal Transformer Encoder**. This component captures the temporal dependencies and aggregates the visual information over the sequence of frames, producing a comprehensive video representation (CLS token).
4.  **Feature Projection:** The aggregated visual feature is projected into a higher-dimensional space (2048-dim) to align with the embedding space of the DeepSeek language model.
5.  **Multi-modal Input for LLM:** The projected visual embedding is prepended to the token embeddings of a textual prompt. This combined input forms the multi-modal input for the DeepSeek model.
6.  **Commentary Generation (DeepSeek):** The **DeepSeek-Coder-1.3B-Instruct** model, fine-tuned on cricket commentary data, generates a raw commentary text based on the visual and textual input.
7.  **Commentary Summarization (Groq API):** The raw commentary is sent to the **Groq API (Llama 3.1 8B Instant)** with a highly engineered prompt to refine it into a concise, professional, and broadcast-style single sentence.
8.  **Speech Synthesis (ElevenLabs API):** The summarized commentary is converted into high-quality speech using the **ElevenLabs API**.
9.  **Audio-Video Mixing:** The synthesized speech is mixed with ambient stadium sounds and then overlaid onto the original video, resulting in the final commentary-rich video output.

Input Video
↓
Frame Extraction & Preprocessing (CLIP preprocess)
↓
Visual Feature Extraction (CLIP ViT-B/32, Frozen)
↓
Temporal Feature Aggregation (Temporal Transformer Encoder)
↓
Feature Projection (Linear Layers + GELU + LayerNorm + Tanh)
↓
Multi-modal Input (Visual Embeddings + Text Prompt Embeddings)
↓
Commentary Generation (DeepSeek-Coder-1.3B-Instruct, Fine-tuned)
↓
Raw Commentary
↓
Commentary Summarization (Groq API - Llama 3.1 8B Instant)
↓
Cleaned & Concise Commentary
↓
Text-to-Speech (ElevenLabs API)
↓
Synthesized Speech
↓
Audio Mixing (Synthesized Speech + Stadium Ambience)
↓
Final Commentary Video


---

## 🗂 Project Structure

Cricket-Commentary/
├── assets/
│   └── Stadium_Ambience.mp3    # Crowd sound for audio mixing
├── final_data/
│   └── Data_updated_1.json     # Annotated dataset for training
├── .gitignore
├── README.md                   # Project documentation (this file)
├── app.py                      # Gradio interface for demonstration
├── inference.py                # Core inference pipeline for commentary generation
├── requirement.txt             # Python dependencies
└── Training.ipynb              # Jupyter notebook for model training (renamed from fork-of-train-commentary)


---

## 🧩 Key Components Explained

### 1. `CricketCommentaryDataset` (in `Training.ipynb`)
* **Purpose:** Custom PyTorch dataset for loading video frames and corresponding prompts/responses.
* **Frame Extraction:** Extracts `num_frames` from a specified video segment (`start_time` to `end_time`), applying action-focused cropping.
* **CLIP Integration:** Uses CLIP's `preprocess` function to prepare images for feature extraction.

### 2. `TemporalTransformerEncoder` (in `Training.ipynb` and `inference.py`)
* **Purpose:** Learns temporal relationships between visual features extracted from video frames.
* **Architecture:** A standard Transformer Encoder with learnable `cls_token` and `position_embed` for sequence representation.
* **Output:** Provides a consolidated `cls` token embedding representing the entire sequence, and individual `tokens` embeddings.

### 3. `CricketCommentator` (in `Training.ipynb` and `inference.py`)
* **Purpose:** The main multi-modal model integrating CLIP, a temporal encoder, and DeepSeek.
* **CLIP Backbone:** Loads a frozen CLIP `ViT-B/32` for robust visual feature extraction.
* **Projection Layer:** A `Sequential` block that maps the 512-dim temporal encoder output to the 2048-dim input expected by DeepSeek.
* **DeepSeek Integration:** Loads `deepseek-ai/deepseek-coder-1.3b-instruct` and its tokenizer. For training, it selectively unfreezes the last few layers and the head for fine-tuning.
* **`forward` method:** Processes frames through CLIP and the temporal encoder, projects the visual embeddings, and returns them for concatenation with text embeddings.
* **`compute_loss` (Training):** Calculates the loss for training by combining visual and text embeddings, creating a combined attention mask, and masking prompt tokens for loss calculation.
* **`generate_commentary` (Inference):** Extracts frames, passes them through the visual pipeline, constructs a multi-modal input with a detailed prompt, and uses DeepSeek's `generate` method to produce raw commentary.

### 4. `train_model` (in `Training.ipynb`)
* **Purpose:** Orchestrates the training process.
* **Optimizer:** `AdamW` with separate learning rates for temporal/projection layers and DeepSeek's trainable layers.
* **Schedulers:** `ReduceLROnPlateau` for validation loss and `get_linear_schedule_with_warmup` for overall training.
* **Training Loops:** Iterates through epochs, calculates loss, and applies gradient accumulation.
* **Mixed Precision:** Uses `torch.cuda.amp.GradScaler` for efficient mixed-precision training.
* **Early Stopping:** Monitors validation loss to prevent overfitting.

### 5. Inference Pipeline (`inference.py`)
* **`summarize_commentary`:** Communicates with the Groq API to refine the raw commentary.
* **`text_to_speech`:** Interacts with the ElevenLabs API to convert text to audio.
* **`mix_audio`:** Uses `moviepy` and `pydub` to combine the generated speech with background crowd noise and integrate it into the original video.
* **`main` function:** The orchestrator for the entire inference process, from loading the model weights (from Hugging Face Hub) to generating the final video.

---

## 🔁 Training Pipeline

The training pipeline is designed for robustness and efficiency:

1.  **Data Preparation:**
    * Loads annotated cricket video data from `final_data/Data_updated_1.json`.
    * Splits data into 85% training and 15% validation sets.
    * `CricketCommentaryDataset` handles video frame extraction, cropping, and CLIP preprocessing.
    * `DataLoader` manages batching, with a small `batch_size=2` due to memory constraints, and `num_workers=2` for parallel data loading.
2.  **Model Initialization:**
    * `CricketCommentator` model is initialized in `train_mode=True`.
    * CLIP parameters are frozen to leverage pre-trained visual knowledge.
    * The last few transformer blocks, norm layer, and language model head of DeepSeek are unfrozen for fine-tuning, allowing the model to adapt to commentary generation while retaining its strong language capabilities.
3.  **Optimization:**
    * **AdamW optimizer** is used.
    * Different learning rates are applied to the newly trained temporal/projection layers and the fine-tuned DeepSeek layers (DeepSeek layers get a smaller LR).
    * **Gradient Accumulation** (`accum_steps=4`) is implemented to effectively simulate larger batch sizes, which is crucial given memory limitations with large language models.
    * **Mixed Precision Training (AMP)** is enabled using `torch.cuda.amp.GradScaler` to speed up training and reduce memory usage.
4.  **Learning Rate Scheduling:**
    * A **linear learning rate scheduler with a warm-up phase** is used to gradually increase the learning rate at the beginning of training, preventing instability.
    * `ReduceLROnPlateau` monitors validation loss and reduces the learning rate if no improvement is observed, aiding convergence.
5.  **Training & Validation:**
    * The model is trained for 30 epochs, with validation performed after each epoch.
    * Training loss and validation loss are tracked.
    * **Early Stopping** with a `patience=5` prevents overfitting by stopping training if validation loss does not improve for a set number of epochs.
    * The `best_model.pth` (which is later renamed to `best_model_1.pth` for Hugging Face upload) is saved based on the lowest validation loss.

---

## 🚀 Inference Pipeline

The inference pipeline takes a video and outputs a new video with AI-generated commentary:

1.  **Model Loading:** The pre-trained `CricketCommentator` model weights are downloaded from the Hugging Face Hub (`switin06/Deepseek_Cricket_commentator/best_model_1.pth`).
2.  **Frame Extraction:** The input video is processed to extract a fixed number of representative frames.
3.  **Visual Embedding Generation:** These frames are passed through the CLIP encoder, temporal transformer, and projection layers to generate a single visual embedding representing the video.
4.  **Commentary Generation:** The visual embedding is concatenated with a carefully crafted textual prompt (e.g., "Provide a sequential description of the cricket delivery...") and fed into the DeepSeek model. The model generates raw, detailed commentary.
5.  **Commentary Summarization:** The raw commentary is sent to the **Groq API** (`llama-3.1-8b-instant` model) with a specific prompt to condense it into a single, broadcast-ready sentence. This step ensures conciseness and professionalism.
6.  **Text-to-Speech:** The summarized commentary is converted into an audio file using the **ElevenLabs API**, utilizing a selected voice for realistic narration.
7.  **Audio Mixing:** The generated speech audio is then mixed with a pre-recorded stadium ambiance sound (`Stadium_Ambience.mp3`). The crowd sound is adjusted for volume and length to blend naturally with the commentary.
8.  **Final Video Production:** The mixed audio track is combined with the original video, resulting in a new video file (`final_video.mp4`) that includes the AI-generated commentary.

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.8+
* NVIDIA GPU (CUDA recommended for training and faster inference)

### 1. Clone the repository

```bash
git clone [https://github.com/switin06/Cricket-Commentary.git](https://github.com/switin06/Cricket-Commentary.git)
cd Cricket-Commentary
2. Create and activate a virtual environment (recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
3. Install dependencies
Bash

pip install -r requirement.txt
4. Set up API Keys
This project uses external APIs for summarization (Groq) and Text-to-Speech (ElevenLabs). You'll need to obtain API keys for both.

Groq API Key: https://console.groq.com/keys

ElevenLabs API Key: https://elevenlabs.io/

Create a .env file in the root directory of the project and add your keys:

GROQ_API_KEY="your_groq_api_key_here"
ElevenLabs="your_elevenlabs_api_key_here"
🚀 Usage
1. Training the Model (Optional)
If you wish to re-train the model or fine-tune it further, follow these steps:

Prepare Data: Ensure Data_updated_1.json is in the final_data/ directory.

Run Training Notebook:

Bash

jupyter notebook Training.ipynb
Execute all cells in the notebook. The best model weights will be saved as best_model.pth.

2. Generating Commentary using the Gradio Interface
The easiest way to use the system is through its interactive Gradio interface.

Bash

python app.py
This will launch a web interface in your browser (usually at http://127.0.0.1:7860).

Upload Video: Click on the "Upload Cricket Video" box and select your .mp4 cricket video file.

Generate: The system will automatically process the video and generate the commentary.

View Output: The "Generated Commentary Video" box will display the video with the AI-powered commentary and crowd ambiance.

3. Running Inference Manually (Developer)
For developers, you can also run the inference pipeline directly from the inference.py script.

Python

# Example of how to use main function in inference.py
from inference import main

video_file_path = "path/to/your/cricket_video.mp4"
output_video = main(video_file_path)
print(f"Generated video with commentary saved at: {output_video}")
📈 Results & Demos
The system consistently generates coherent and contextually relevant cricket commentary. The integration of Groq's summarization ensures the commentary is concise and suitable for broadcast. ElevenLabs provides high-quality speech, making the output highly professional.

Test Accuracy (Training): The model achieves a good validation loss, indicating effective learning of video-text relationships for commentary generation.

Qualitative Results: The generated commentary accurately describes the bowler's action, the batsman's response, and the outcome of the ball, often in a single, well-phrased sentence.

(Add screenshots or links to demo videos of the generated commentary here if available for a live project)

📦 Model Weights
The pre-trained model weights for the CricketCommentator are hosted on Hugging Face:

Repository: switin06/Deepseek_Cricket_commentator

File: best_model_1.pth

The inference.py script automatically downloads these weights when the application runs.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing
Contributions are welcome! If you have ideas for improvements or new features, please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix:

Bash

git checkout -b feature/your-new-feature
Make your changes, ensuring you follow the existing code style and add appropriate documentation.

Test your changes thoroughly.

Commit your changes with a clear and concise message.

Push your branch to your forked repository.

Open a Pull Request to the main branch of this repository, describing your changes and their benefits.

Ideas for Future Enhancements:
Event Detection: Implement more granular event detection (e.g., boundaries, wickets, specific shot types) to drive commentary.

Multi-Lingual Commentary: Extend the system to generate commentary in multiple languages.

Real-time Processing: Optimize the pipeline for near real-time commentary generation during live streams.

User Feedback Integration: Allow users to provide feedback on generated commentary to further refine the model.

Diverse Commentary Styles: Explore generating commentary in different styles (e.g., analytical, enthusiastic, sarcastic).

⭐ If this project helps you, consider giving it a star!
