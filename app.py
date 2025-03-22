import gradio as gr
from transformers import pipeline
import torch

# Load Gemma model (2b-it for lighter compute; swap to 7b-it if you have GPU)
generator = pipeline("text-generation", model="google/gemma-2b-it", device=0 if torch.cuda.is_available() else -1)

def generate_text(prompt, max_length):
    """Generate text from a prompt with specified length."""
    try:
        result = generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)
        # Convert tensor output to string if needed
        if hasattr(result[0]["generated_text"], 'tolist'):
            return result[0]["generated_text"].tolist()
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Define the UI
with gr.Blocks(title="Gemma Text Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Gemma Text Generator
        Unleash creativity with Google's Gemma model! Enter a prompt below, adjust the length, and watch the magic happen.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Your Prompt", placeholder="e.g., 'Once upon a time...'")
            length_slider = gr.Slider(20, 200, value=50, step=10, label="Max Output Length")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output_text = gr.Textbox(label="Generated Text", lines=5, interactive=False)

    # Examples for quick testing
    examples = gr.Examples(
        examples=["Write a haiku about the moon", "Tell me a short sci-fi story", "Compose a funny tweet"],
        inputs=[prompt_input]
    )
    
    # Button click action
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, length_slider],
        outputs=output_text
    )

    # Footer
    gr.Markdown("Built with Streamlit using Gemma by Google | Deployed via Gradio")

# Launch the app
demo.launch()