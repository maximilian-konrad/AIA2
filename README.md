# Automated-Image-Analysis (AIA)

This repository contains the code and data for the AIA2 project. The goal of the project is to create a simple Image Analytics Pipeline that can be used in future research.

## Setup

### Conda Environment

To set up the Conda environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/maximilian-konrad/AIA2.git
   cd automated-image-analytics
   ```

2. **Create the Conda environment**:

    Ensure you have Conda installed. Then, create the environment using the provided environment.yml file:
    
    ```bash
    conda env create -f env.yml
    ```

3. **Activate the environment**:

    ```bash
    conda activate aia
    ```

### Run the pipeline

Option A: **Use the webapp**:
    Run app.py in ./

    Open http://127.0.0.1:5000 and upload your images. 
    Once completed, the results will be available for download.

Option B: **Use the Jupyter notebook**:

    Run pipeline.ipynb in ./src/notebooks/

    Results will be stored as .XLSX file in ./outputs/