import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from ..utils.helper_functions import load_config

def detect_objects(df_images):

    # Load the full configuration directly from params.yaml
    full_config = load_config()
    objects_to_detect = full_config.get("features", {}).get("detect_objects", {}).get("parameters", {}).get("objects_to_detect", [])
    print(f"Objects to detect: {objects_to_detect}")

    df = df_images.copy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        image = Image.open(image_path).convert("RGB")
        prompt = "<OD>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"] if "input_ids" in inputs else None,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task="<OD>", 
            image_size=(image.width, image.height)
        )
    
        detected_objects = parsed_answer.get('<OD>', {}).get('labels', [])
        detected_objects_lower = [obj.lower() for obj in detected_objects]
        
        for obj in objects_to_detect:
            column_name = f"contains_{obj.lower()}"
            df.loc[idx, column_name] = obj.lower() in detected_objects_lower
    
    return df