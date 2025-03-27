import gradio as gr
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import torch

models = {
    "MeinaMixV12": "/Users/eximias/stable-diffusion-webui/converted_models/meinamix_v12Final",
    "ToonYou Beta6": "/Users/eximias/stable-diffusion-webui/converted_models/toonyou_beta6_diffusers"
} 

negative_prompts_preset = {
    "Mặc định": "(ugly, deformed, worst quality, low quality, letterboxed)",
    "Chân thực hơn": "(lowres, bad anatomy, bad hands, missing fingers, blurry, distorted)",
    "Tránh hiệu ứng lạ": "(text, watermark, signature, artifacts, distorted edges)",
    "Tránh ảnh chụp, 3D, NSFW": "(realistic, photorealistic, photograph, 3D, NSFW, (worst quality, low quality:1.3))"
}

image_sizes = [384, 512, 640, 768, 1024]

guidance_scales = [7, 7.5, 8, 8.5, 9]

inference_steps = [20, 25, 30, 35]

schedulers = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler
}

current_model = list(models.values())[0]
pipeline = DiffusionPipeline.from_pretrained(
    current_model, use_safetensors=True, safety_checker=None, requires_safety_checker=False
)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline.to(device)

def load_model(model_name, scheduler_name):
    global pipeline
    model_path = models[model_name]
    pipeline = DiffusionPipeline.from_pretrained(
        model_path, use_safetensors=True, safety_checker=None, requires_safety_checker=False
    )
    pipeline.scheduler = schedulers[scheduler_name].from_config(pipeline.scheduler.config)
    pipeline.to(device)
    return f"Đã tải mô hình: {model_name} với Scheduler: {scheduler_name}"

# Hàm cập nhật negative prompt khi chọn preset
def update_negative_prompt(selected_preset):
    return negative_prompts_preset[selected_preset]

# Hàm cập nhật Scheduler
def update_scheduler(scheduler_name):
    global pipeline
    pipeline.scheduler = schedulers[scheduler_name].from_config(pipeline.scheduler.config)
    return f"Đã thay đổi Scheduler thành: {scheduler_name}"

# Hàm tạo ảnh
def generate_image(prompt, negative_prompt, width, height, guidance_scale, num_steps):
    generated_images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps
    ).images
    return generated_images[0]

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 TramTram - Tạo ảnh với Gradio")

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()), value=list(models.keys())[0], label="Chọn mô hình"
            )
            scheduler_dropdown = gr.Dropdown(
                choices=list(schedulers.keys()), value="Euler", label="Chọn Scheduler"
            )
            model_status = gr.Textbox(label="Trạng thái mô hình", interactive=False)

            prompt = gr.Textbox(label="Prompt", value="(masterpiece, best quality), ", placeholder="Nhập prompt của bạn")

            negative_preset_dropdown = gr.Dropdown(
                choices=list(negative_prompts_preset.keys()),
                value="Mặc định",
                label="Chọn Negative Prompt mẫu"
            )
            negative_prompt = gr.Textbox(label="Negative Prompt", value=negative_prompts_preset["Mặc định"], placeholder="Hoặc nhập Negative Prompt")

            with gr.Row():
                width_dropdown = gr.Dropdown(choices=image_sizes, value=512, label="Chiều rộng")
                height_dropdown = gr.Dropdown(choices=image_sizes, value=768, label="Chiều cao")
                guidance_dropdown = gr.Dropdown(choices=guidance_scales, value=7.5, label="Guidance")
                steps_dropdown = gr.Dropdown(choices=inference_steps, value=30, label="Steps")


            generate_button = gr.Button("Tạo ảnh")

        with gr.Column():
            image_output = gr.Image(label="Ảnh đã tạo", height=512, width=512)

    # Khi chọn model hoặc scheduler, tải lại model và cập nhật scheduler
    model_dropdown.change(fn=load_model, inputs=[model_dropdown, scheduler_dropdown], outputs=[model_status])
    scheduler_dropdown.change(fn=update_scheduler, inputs=[scheduler_dropdown], outputs=[model_status])

    # Khi chọn negative prompt mẫu, cập nhật textbox
    negative_preset_dropdown.change(fn=update_negative_prompt, inputs=[negative_preset_dropdown], outputs=[negative_prompt])

    # Khi bấm nút "Tạo ảnh", tạo ảnh từ model hiện tại với tham số tuỳ chỉnh
    generate_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, width_dropdown, height_dropdown, guidance_dropdown, steps_dropdown],
        outputs=image_output
    )

# Chạy giao diện
demo.launch()
