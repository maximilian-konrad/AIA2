# Automated-Image-Analysis (AIA)

This repository contains the code and data for the AIA2 project. The goal of the project is to create a simple Image Analytics Pipeline that can be used in future research.

## ToDo

- [ ] Fix issue with double-implementation of clarity (clarity1 != clarity2)
- [ ] Add a webapp
- [ ] Add a Docker container

## Setup

### Using a Conda Environment

To set up the Conda environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/maximilian-konrad/AIA2.git
   cd AIA2
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

### Using a Python Virtual Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/maximilian-konrad/AIA2.git
   cd AIA2
   ```

2. **Create a virtual environment**:

   ```
   python -m venv aiaenv
   ```

   On Linux/Mac:

   ```
   source aiaenv/bin/activate
   ```

   On Windows (cmd):

   ```
   aiaenv\Scripts\activate
   ```

3. **Install requirements**:

   ```
   pip install -r requirements.txt
   ```

### Using Docker

1. **Build the image**:

   ```
   sudo docker build -t aia2 .
   ```

2. **Start the container**:

   Once the build is complete, run

   ```
   sudo docker run -p 8888:8888 -v $(pwd):/app aia2
   ```

3. **Log in to the Jupyter Server**:

   The Jupyter server will start and a token will appear in your terminal.

   In your browser, open http://localhost:8888/tree

   You will be asked for a password/token for authentication.
   Paste the token in the prompt to continue to the application.

### Select features and parameters

The features and parameters are defined in the configuration.yaml file.
This file is stored in the config folder.

To enable or disable a feature, set the resepective parameter `active` to True or False.

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
