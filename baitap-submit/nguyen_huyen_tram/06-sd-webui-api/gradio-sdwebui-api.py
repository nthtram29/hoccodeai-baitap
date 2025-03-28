import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

URL = "http://127.0.0.1:7860"

negative_prompts_preset = {
    "Mặc định": "(ugly, deformed, worst quality, low quality, letterboxed)",
    "Chân thực hơn": "(lowres, bad anatomy, bad hands, missing fingers, blurry, distorted)",
    "Tránh hiệu ứng lạ": "(text, watermark, signature, artifacts, distorted edges)",
    "Tránh ảnh chụp, 3D, NSFW": "(realistic, photorealistic, photograph, 3D, NSFW, (worst quality, low quality:1.3))"
}

samplers = [
    "Euler", "Euler a", "LMS", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE",
    "DPM fast", "DPM adaptive", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM", "PLMS"
]

def fetch_models():
    response = requests.get(f"{URL}/sdapi/v1/sd-models")
    if response.status_code == 200:
        models = response.json()
        return {model["model_name"]: model["model_name"] for model in models}
    return {}

def set_model(model_name):
    payload = {"sd_model_checkpoint": model_name}
    response = requests.post(f"{URL}/sdapi/v1/options", json=payload)
    return f"Đã chọn mô hình: {model_name}" if response.status_code == 200 else "Lỗi khi chọn mô hình"

def update_negative_prompt(selected_preset):
    return negative_prompts_preset[selected_preset]

def generate_image(prompt, negative_prompt, sampler_index, width, height, guidance_scale, steps, seed):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_index": sampler_index,
        "width": width,
        "height": height,
        "cfg_scale": guidance_scale,
        "steps": steps,
        "seed": int(seed) if seed.isdigit() else -1,
    }
    response = requests.post(f"{URL}/sdapi/v1/txt2img", json=payload)
    
    if response.status_code == 200:
        img_data = response.json()["images"][0]
        
        image_bytes = base64.b64decode(img_data)
        image = Image.open(BytesIO(image_bytes))
        
        return image 
    return "Lỗi khi tạo ảnh"

models = fetch_models()

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 TramTram - Tạo ảnh với Gradio")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(choices=list(models.keys()), value=list(models.keys())[0] if models else "", label="Chọn mô hình")
            model_status = gr.Textbox(label="Trạng thái mô hình", interactive=False)
            
            prompt = gr.Textbox(label="Prompt", placeholder="Nhập prompt của bạn")
            
            negative_preset_dropdown = gr.Dropdown(
                choices=list(negative_prompts_preset.keys()),
                value="Mặc định",
                label="Negative Preset",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt", 
                value=negative_prompts_preset["Mặc định"], 
                placeholder="Hoặc nhập Negative Prompt",
            )
            
            # Hàng Width & Height
            with gr.Row():
                width_dropdown = gr.Slider(384, 1024, step=8, value=512, label="Chiều rộng")
                height_dropdown = gr.Slider(384, 1024, step=8, value=768, label="Chiều cao")
                
            with gr.Row():
                guidance_slider = gr.Slider(1, 20, step=0.5, value=7.5, label="Guidance Scale")
                steps_slider = gr.Slider(10, 50, step=5, value=30, label="Steps")

            with gr.Row():
                seed_input = gr.Textbox(label="Seed", value="-1", placeholder="Nhập seed (-1 = random)")
                sampler_dropdown = gr.Dropdown(choices=samplers, value="Euler", label="Sampler")
                
            generate_button = gr.Button("Tạo ảnh")
        
        with gr.Column():
            image_output = gr.Image(label="Ảnh đã tạo")
    
    model_dropdown.change(fn=set_model, inputs=[model_dropdown], outputs=[model_status])

    negative_preset_dropdown.change(fn=update_negative_prompt, inputs=[negative_preset_dropdown], outputs=[negative_prompt])

    generate_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, sampler_dropdown, width_dropdown, height_dropdown, guidance_slider, steps_slider, seed_input],
        outputs=image_output
    )

demo.launch()
