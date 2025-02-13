

def calculate_aesthetic_scores(df_images):
    """
    Calculates the aesthetic scores for a list of images using the NIMA model.

    :param df_images: DataFrame with a column 'filename' containing paths to image files.
    :param model_path: Path to the pre-trained NIMA model.
    :return: DataFrame with an additional column 'aesthetic_score'.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    ### TODO: If not downloaded already, download weights from https://github.com/idealo/image-quality-assessment/ and save to AIA2/src/AIA/weights/
    # You can use the helper function download_weights() to download the weights

    ### TODO: Load model from weights from AIA2/src/AIA/weights/

    # Process each image in the dataframe
    for idx, image_path in enumerate(tqdm(df_images['filename'])):          
        try:
            ### TODO: Call predict() from NIMA repository and get an aesthetic_score for each image

            # Update the DataFrame
            df.loc[idx, 'nima_score'] = aesthetic_score
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            df.loc[idx, 'nima_score'] = f"Error: {str(e)}"
       
    return df



    





        
