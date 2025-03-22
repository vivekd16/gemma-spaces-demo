import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="google/gemma-2b-it")

def generate_text(prompt, max_length):
    return generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)[0]["generated_text"]

with gr.Blocks(title="Gemma Text Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Gemma Text Generator\nType a prompt and let Gemma weave some magic!")
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Your Prompt", placeholder="e.g., 'Once upon a time...'")
            length_slider = gr.Slider(20, 200, value=50, step=10, label="Max Length")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output_text = gr.Textbox(label="Generated Text", lines=5, interactive=False)
    gr.Examples(examples=["Write a haiku", "Tell a sci-fi tale"], inputs=[prompt_input])
    generate_btn.click(fn=generate_text, inputs=[prompt_input, length_slider], outputs=output_text)
    gr.Markdown("Built by [vivekd16](https://github.com/vivekd16) | Powered by Gemma")

demo.launch()
