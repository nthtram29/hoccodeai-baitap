import base64
import requests

URL = "http://127.0.0.1:7860"

def set_model():
    option_payload = {
        "sd_model_checkpoint": "toonyou_beta6.safetensors"
    }
    response = requests.post(f"{URL}/sdapi/v1/options", json=option_payload)

    if response.status_code == 200:
        print("✅ Model switched to 'toonyou_beta6.safetensors'")
    else:
        print("❌ Failed to switch model:", response.status_code)

def base64_to_image(base64_string, save_path='output_image.png'):
    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(base64_string))

def text_to_image(prompt, width=512, height=512):
    print("🎨 Starting Inference...")

    payload = {
        "prompt": prompt,
        "negative_prompt": "worst quality, low quality, watermark, text, error, blurry, jpeg artifacts, cropped, jpeg artifacts, signature, watermark, username, artist name, bad anatomy",
        "steps": 30,
        "cfg_scale": 7.5,
        "sampler_index": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
        "width": width,
        "height": height,
    }

    response = requests.post(f"{URL}/sdapi/v1/txt2img", json=payload)
    resp_json = response.json()

    print("✅ Inference Completed!")

    for i, img in enumerate(resp_json['images']):
        save_path = f"output_image_{i}.png"
        print(f"💾 Saving image: {save_path}")
        base64_to_image(img, save_path)


if __name__ == "__main__":
    set_model()
    user_prompt = input("Prompt: ")
    user_width = int(input("Width(px): "))
    user_height = int(input("Height(px): "))

    text_to_image(user_prompt, user_width, user_height)
