import torch
import math
import cv2
import json
import time
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
import clip
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from IPython.display import Audio
import re
from groq import Groq
from moviepy.editor import VideoFileClip, AudioFileClip,CompositeAudioClip
from pydub import AudioSegment
import shutil
from google.colab import userdata
import gradio as gr
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_frames, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.position_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))
        nn.init.trunc_normal_(self.position_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embed[:, :x.size(1)]
        x = self.transformer(x)
        return {
            "cls": x[:, 0],
            "tokens": x[:, 1:]
        }
class CricketCommentator(nn.Module):
    def __init__(self, train_mode=False, num_frames=16, train_layers=2):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_frames = num_frames

        import clip
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip = self.clip.float()

        if train_mode:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.temporal_encoder = TemporalTransformerEncoder(
            embed_dim=512,
            num_heads=8,
            num_layers=3,
            num_frames=num_frames,
            dropout=0.1
        ).to(self.device).float()

        # Updated projection for DeepSeek (2048-dim)
        self.projection = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.Tanh()
        ).to(self.device).float()

        # DeepSeek model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct").to(self.device).float()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N layers if training
        if train_mode and train_layers > 0:
            # Unfreeze last transformer blocks
            for block in self.model.model.layers[-train_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Unfreeze final norm and head
            for param in self.model.model.norm.parameters():
                param.requires_grad = True
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

    def forward(self, frames):
        batch_size = frames.shape[0]
        frames = frames.view(-1, 3, 224, 224)
        with torch.no_grad():
            frame_features = self.clip.encode_image(frames.to(self.device))
        frame_features = frame_features.view(batch_size, self.num_frames, -1).float()
        frame_features = F.normalize(frame_features, p=2, dim=-1)

        temporal_out = self.temporal_encoder(frame_features)
        visual_embeds = self.projection(temporal_out["cls"])
        return F.normalize(visual_embeds, p=2, dim=-1).unsqueeze(1)


    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total_frames // self.num_frames)
        frames = []

        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                crop_size = min(h, w) // 2
                y, x = (h - crop_size) // 2, (w - crop_size) // 2
                cropped = cv2.cvtColor(frame[y:y+crop_size, x:x+crop_size], cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cropped)
                frames.append(self.preprocess(pil_image))
                if len(frames) >= self.num_frames:
                    break
            else:
                break
        cap.release()
        if len(frames) < self.num_frames:
            frames.extend([torch.zeros(3, 224, 224)] * (self.num_frames - len(frames)))
        return torch.stack(frames)
    def generate_commentary(self, video_path):
        frames = self.extract_frames(video_path).unsqueeze(0).to(self.device)
        visual_embeds = self.forward(frames)  # Shape: [1, 1, 2560]

        # Prepare text prompt
        prompt = ("USER: <video> Provide a sequential description of the cricket delivery in the video. Start with the bowler's run-up, then describe the delivery, the batsman's action, and finally the outcome of the ball. Keep it concise also make sure that you won't cross 2 lines and the commentary must be in a professional tone.ASSISTANT:")

        # Tokenize text prompt
        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512).to(self.device)

        # Get token embeddings
        token_embeds = self.model.model.embed_tokens(inputs['input_ids'])

        # Combine visual and text embeddings
        inputs_embeds = torch.cat([visual_embeds, token_embeds], dim=1)

        # Create attention mask (1 for visual token + text tokens)
        attention_mask = torch.cat([
            torch.ones(visual_embeds.shape[:2], dtype=torch.long).to(self.device),
            inputs['attention_mask']
        ], dim=1)

        # Generate commentary
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            min_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract and clean generated text
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        commentary = full_text.split("ASSISTANT:")[-1].strip()
        print(commentary)
        return commentary



# -------------------- PIPELINE --------------------

def summarize_commentary(commentary, client):
  prompt = f"""
🔧 Final Optimized Prompt (Prompt-Engineered):

You are a professional cricket commentary editor.
You will receive raw cricket commentary text. It may contain grammar errors, unclear phrasing, or inconsistent tone.

Your task is to:

Rewrite it as a concise, broadcast-style commentary (1 sentence max).

Keep it short and direct—describe the action and the outcome only. No exaggeration or filler.

If no boundary (four/six) is mentioned, do not add one.

Maintain proper punctuation and clarity for TTS (Text-to-Speech) delivery.

Output only the cleaned commentary. No extra text.

Input:
{commentary}

Output:
"""




  chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )
  print("="*50)
  print(chat_completion.choices[0].message.content.strip())
  print("="*50)
  return chat_completion.choices[0].message.content.strip()

def text_to_speech(text, elevenlabs_api_key, output_path):
    elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)
    audio_stream = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="URgDTjqBVr48zeu6FETI",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
def mix_audio(video_path, voice_path, crowd_path, output_path):
    video = VideoFileClip(video_path)
    video_duration_ms = video.duration * 1000
    voice = AudioSegment.from_file(voice_path)[:int(video_duration_ms - 100)]
    crowd = AudioSegment.from_file(crowd_path) - 10
    while len(crowd) < len(voice):
        crowd += crowd
    crowd = crowd[:len(voice)]
    mixed = crowd.overlay(voice)

    crowd_head = AudioSegment.from_file(crowd_path) - 15
    while len(crowd_head) < (video_duration_ms - len(mixed)):
        crowd_head += crowd_head
    crowd_head = crowd_head[:int(video_duration_ms - len(mixed))]

    final_audio = crowd_head + mixed
    temp_audio_path = "temp_mixed_audio.mp3"
    final_audio.export(temp_audio_path, format="mp3")

    final_video = video.set_audio(AudioFileClip(temp_audio_path))
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
def main(video_path):
    load_dotenv()

    model_weights_path = "/content/drive/MyDrive/InferenceData/best_model_1.pth"
    crowd_path = "/content/drive/MyDrive/InferenceData/Stadium_Ambience.mp3"

    # Load model
    model = CricketCommentator(train_mode=False)
    model.load_state_dict(torch.load(model_weights_path, map_location=model.device))
    model.eval()

    # Generate raw commentary
    raw_commentary = model.generate_commentary(video_path)

    # Summarize using Groq API
    client = Groq(api_key=userdata.get("GROQ_API_KEY"))
    clean_commentary = summarize_commentary(raw_commentary, client)

    # Text to speech
    tts_path = "commentary_final.mp3"
    text_to_speech(clean_commentary, userdata.get("ElevenLabs"), tts_path)

    short_audio_path = "pro_audio3.mp3"
    os.system(f"ffmpeg -y -i {tts_path} -ss 0 -t 3 {short_audio_path}")

    # Final video output
    output_video_path = "final_video.mp4"
    mix_audio(video_path, short_audio_path, crowd_path, output_video_path)

    return output_video_path
