from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import os

def detect_objects(self, df_images):
    """
    Detect objects in images using a pre-trained model.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added columns indicating the presence of detected objects
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters
        objects_to_detect = self.config.get("features", {}).get("detect_objects", {}).get("parameters", {}).get("objects_to_detect", [])
        print(f"Objects to detect: {objects_to_detect}")

        # Initialize model
        try:
            model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(self.device)
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return df

        # Initialize columns
        for obj in objects_to_detect:
            column_name = f"contains_{obj.lower()}"
            df[column_name] = False

        # Process images
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Prepare inputs
                prompt = "<OD>"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)

                # Generate predictions
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"] if "input_ids" in inputs else None,
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False
                )

                # Process results
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

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_object_detection'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in object detection setup: {str(e)}")
        return df