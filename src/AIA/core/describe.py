"""
Add general comments here.
"""

# Imports
import os
import io
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

def describe_blip(self, df_images): 
    """
    This function generates a textual description of an image using the BLIP model.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize new columns with NaN
    df['descrBlip'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        df.loc[idx, 'descrBlip'] = caption

    return df

def describe_llm(self, df_images, prompt="Describe the image."): 
    """
    This function generates a textual description of an image using a large language model (Phi-4-multimodal-instruct).
    The prompt to generate the description can be customized.
    Note: Requires large GPU VRAM (> 16GB).

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Define model path
    model_path = "microsoft/Phi-4-multimodal-instruct"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='eager',
    ).cuda()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>' 
    combined_prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'
    
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize new columns with NaN
    df['descrLLM'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        image = Image.open(image_path)
        inputs = processor(text=combined_prompt, images=image, return_tensors='pt').to('cuda:0')
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Store computed features in DataFrame
        df.loc[idx, 'descrLLM'] = response

    return df

