from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import torch

# Load model
pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True, safety_checker=None, requires_safety_checker=False
)

# Luôn sử dụng CPU trên Mac
device = "cpu"
dtype = torch.float32
pipeline.to(device, dtype=dtype)

# Thiết lập mặc định
DEFAULT_NEGATIVE_PROMPT = "ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality"
DEFAULT_ADDITIONAL_PROMPT = "highly detailed, 8K, best quality"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7
DEFAULT_SCHEDULER = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

pipeline.scheduler = DEFAULT_SCHEDULER

print("Pipeline đã tải xong!")

# Nhập thông tin từ người dùng
prompt = input("Prompt: ")
height = int(input("Height(là bội số của 8): "))
width = int(input("Width(à bội số của 8): "))

# Bổ sung prompt mặc định
final_prompt = f"{prompt}, {DEFAULT_ADDITIONAL_PROMPT}"

# Tạo ảnh
image = pipeline(
    prompt=final_prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    height=height,
    width=width,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_STEPS
).images[0]

# Lưu ảnh
output_path = "output_image.png"
image.save(output_path)
print(f"Ảnh đã được lưu tại: {output_path}")
