import random

import cv2
import einops
import numpy as np
import PIL.Image
import torch
from pytorch_lightning import seed_everything
import sys, os

# controlnet_root = "/home/papagina/Documents/proj_hair/digital_salon/controlnet/"
controlnet_root = "/disk3/proj_viton/ControlNet-v1-1-nightly"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(controlnet_root)

import config
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *

preprocessor = None

model_name = 'control_v11p_sd15_canny'
model = create_model(f'{controlnet_root}/models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict(f'{controlnet_root}/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'{controlnet_root}/models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor

    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


def controlnet_api(image: np.ndarray, gender: str, hairstyle: str, headpose: str, misc_prompt: str = "", num_samples: int = 1, image_resolution: int = 512, detect_resolution: int = 512, ddim_steps: int = 20, guess_mode: bool = False, strength: float = 1.0, scale: float = 9.0, seed: int = 12345, eta: float = 1.0, low_threshold: int = 100, high_threshold: int = 200):
    prompt = f"a {gender} with {hairstyle} hair, {headpose}, {misc_prompt}, photorealisic, 4k, cinematic quality, beautiful hair, dslr, soft lighting"
    print (prompt)
    
    a_prompt = "best quality"
    n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    out_imgs = process(det="Canny", input_image=image, prompt=prompt, a_prompt=a_prompt, n_prompt=n_prompt,
                       num_samples=num_samples, image_resolution=image_resolution, detect_resolution=detect_resolution,
                       ddim_steps=ddim_steps, guess_mode=guess_mode, strength=strength, scale=scale, seed=seed, eta=eta,
                       low_threshold=low_threshold, high_threshold=high_threshold)
    return out_imgs[1:]


def center_crop(img, target_width, target_height):
    """Crops an image to the specified dimensions, keeping the center intact."""

    height, width = img.shape[:2]

    # Calculate the starting coordinates for the crop
    x_start = int((width - target_width) / 2)
    y_start = int((height - target_height) / 2)

    # Crop the image
    cropped_img = img[y_start:y_start + target_height, x_start:x_start + target_width]

    return cropped_img

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_string>")
        sys.exit(1)

    gender = sys.argv[1]
    hairstyle=sys.argv[2]
    headpose=sys.argv[3]
    misc=sys.argv[4]

    
    image = PIL.Image.open("../tmp/controlnet/input.png")
    image = np.asarray(image)
    print(f"image: {image.shape}")

    w = image.shape[1]
    h = image.shape[0]
    target_size = min(w, h)
    target_size = int(target_size * 0.8)
    image = center_crop(image, target_width=target_size, target_height=target_size)
    # image = image[int(h*0.0):int(h*0.6), int(w*0.3):int(w*0.95)]
    # w= image.shape[1]
    # h=image.shape[0]

    # s=w
    # if(w>h):
    #     s=h
    # s=int(w*0.7)
    # image = image[0:s,0:s]
    # PIL.Image.fromarray(image).save(controlnet_root+"../Digital-Salon/tmp/controlnet/input_crop.png")
    PIL.Image.fromarray(image).save("../tmp/controlnet/input_crop.png")

    
    out_images = controlnet_api(image, gender=gender, hairstyle=hairstyle, headpose=headpose, misc_prompt=misc)
    out_image = out_images[0]
    out_image = PIL.Image.fromarray(out_image)
    # out_image.save(controlnet_root+"../Digital-Salon/tmp/controlnet/output.png")
    out_image.save("../tmp/controlnet/output.png")

    