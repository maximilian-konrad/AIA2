# Automated-Image-Analysis (AIA)

This repository contains the code and data for the AIA2 project. The goal of the project is to create a simple Image Analytics Pipeline that can be used in future research.

## ToDo

- [ ] Fix issue with double-implementation of clarity (clarity1 != clarity2)
- [ ] Add a webapp
- [ ] Add a Docker container

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

### Select features and parameters

The features and parameters are defined in the configuration.yaml file.
This file is stored in the config folder.

To enable or disable a feature, set the resepective parameter  `active` to True or False.

### Run the pipeline

Option A: **Use the Jupyter notebook**:

    Run pipeline.ipynb in ./src/notebooks/

    Results will be stored as .XLSX file in ./outputs/

Option B: **INSTABLE: Use the webapp**:
    Run app.py in ./

    Open http://127.0.0.1:5000 and upload your images. 
    Once completed, the results will be available for download.

## Contributing

If you contributed and made any changes to the environment, please update the env.yml file to allow others to deploy this package easily.
You can do so by running the following command:

```bash
conda env  export --no-builds > env.yml
```

Typically, this will add a system-specific line (e.g., `prefix: C:\Users\user1\.conda\envs\aia`) at the end of the env.yml file.
You can remove this line.

Please ensure to include the updated env.yml in your commit and pull request.


