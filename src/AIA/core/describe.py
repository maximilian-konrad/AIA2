"""
Image description generation using BLIP and LLM models.
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
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters
        processor_path = self.config.get('describe_blip', {}).get('processor', 'Salesforce/blip-image-captioning-base')
        model_path = self.config.get('describe_blip', {}).get('model', 'Salesforce/blip-image-captioning-base')

        # Load processor and model
        try:
            processor = BlipProcessor.from_pretrained(processor_path)
            model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"Error loading BLIP model: {str(e)}")
            return df

        # Initialize new columns with empty string
        df['descrBlip'] = ""

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_blip'] = "File not found"
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_blip'] = f"Image load error: {str(e)}"
                    continue

                # Generate description
                try:
                    inputs = processor(image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                    caption = processor.decode(outputs[0], skip_special_tokens=True)
                    df.loc[idx, 'descrBlip'] = caption
                except Exception as e:
                    error = f"Description generation error: {str(e)}"
                    print(f"Error generating description for {image_path}: {error}")
                    df.loc[idx, 'error_blip'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_blip'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in BLIP setup: {str(e)}")
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
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Define model path and prompts
        model_path = "microsoft/Phi-4-multimodal-instruct"
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>' 
        combined_prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'

        # Load model and processor
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True,
                _attn_implementation='eager'
            ).cuda()
            generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading LLM model: {str(e)}")
            return df

        # Initialize new columns with empty string
        df['descrLLM'] = ""

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_llm'] = "File not found"
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_llm'] = f"Image load error: {str(e)}"
                    continue

                # Generate description
                try:
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
                    df.loc[idx, 'descrLLM'] = response
                except Exception as e:
                    error = f"Description generation error: {str(e)}"
                    print(f"Error generating description for {image_path}: {error}")
                    df.loc[idx, 'error_llm'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_llm'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in LLM setup: {str(e)}")
        return df

