import gradio as gr
from inference import main

iface = gr.Interface(
    fn=main,
    inputs=gr.Video(label="Upload Cricket Video"),
    outputs=gr.Video(label="Generated Commentary Video"),
    title="Cricket Commentary Generator",
    description="Upload a cricket video to generate AI-powered commentary with ambient sound."
)

iface.launch(debug=True)
