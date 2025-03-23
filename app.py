import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
from huggingface_hub import login

# Function to load model and tokenizer with better error handling
def load_model(model_name):
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Cache for loaded models
model_cache = {}

def generate_text(prompt, model_name, temperature, max_length, top_p, hf_token):
    if not prompt.strip():
        return "", "**Error:** Please enter a prompt."
    
    try:
        if hf_token.strip():
            login(token=hf_token)
        
        if model_name not in model_cache:
            model, tokenizer, error = load_model(model_name)
            if error:
                return "", f"**Error:** {error}"
            model_cache[model_name] = (model, tokenizer)
        else:
            model, tokenizer = model_cache[model_name]
        
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        
        metrics = f"""
        **Generation Time:** {time_taken:.2f}s
        **Input Length:** {len(prompt.split())} words
        **Output Length:** {len(generated_text.split())} words
        """
        
        return generated_text, metrics
        
    except Exception as e:
        return "", f"**Error:** {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Enhanced Gemma Text Generator")
    
    with gr.Row():
        with gr.Column(scale=2):
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                placeholder="Enter your HF token",
                type="password"
            )
            
            prompt_input = gr.Textbox(
                label="Your Prompt",
                placeholder="Enter your prompt here...",
                lines=4
            )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["google/gemma-2b", "google/gemma-7b"],
                    value="google/gemma-2b",
                    label="Model"
                )
                
                temperature_slider = gr.Slider(
                    0.1, 2.0, value=0.7,
                    label="Temperature",
                    info="Controls creativity"
                )
            
            with gr.Row():
                length_slider = gr.Slider(
                    64, 1024, value=256,
                    step=32,
                    label="Max Length"
                )
                
                top_p_slider = gr.Slider(
                    0.1, 1.0, value=0.9,
                    label="Top-p"
                )
            
            generate_btn = gr.Button("✨ Generate", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=10,
                show_copy_button=True
            )
            metrics_output = gr.Markdown(label="Metrics")
    
    examples = [
        ["Write a haiku about AI"],
        ["Explain quantum computing to a child"],
        ["Write a short story about robots"],
        ["List 5 ways to learn programming"],
    ]
    gr.Examples(examples=examples, inputs=[prompt_input])
    
    generate_btn.click(
        fn=generate_text,
        inputs=[
            prompt_input,
            model_selector,
            temperature_slider,
            length_slider,
            top_p_slider,
            hf_token
        ],
        outputs=[output_text, metrics_output]
    )
    
    # Add documentation
    gr.Markdown("""
    ### Tips
    - Higher temperature (>1.0) = more creative but less focused
    - Lower temperature (<0.5) = more focused but less creative
    - Adjust top-p to control output diversity
    - Use a GPU for faster generation
    """)

if __name__ == "__main__":
    demo.launch()
