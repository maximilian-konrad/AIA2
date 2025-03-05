import numpy as np
from skimage import io, filters
from skimage.measure import regionprops, label
from tqdm import tqdm

def visual_complexity(df_images):
    df = df_images.copy()
    
    for idx, input_image_path in enumerate(tqdm(df_images['filename'])):
        img = io.imread(input_image_path)
        if len(img.shape) > 2:
            img = np.mean(img, axis=2).astype(np.uint8)

        # alternative for MATLAB's adaptive imbinarize
        thresh = filters.threshold_local(img, block_size=35)
        binary_img = img > thresh
        rp_tot = binary_img.shape[0] * binary_img.shape[1]
        labeled_img = label(binary_img)
        regions = regionprops(labeled_img)
        threshold = rp_tot / 25000
        r_spt = sum(1 for region in regions if region.area > threshold)

        df.loc[idx, 'visualComplexity'] = r_spt
        
    return df