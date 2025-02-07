import os
import json

def load_config():
    """
    Loads the config.json file from the project root (assumes the project root is one level up from the script directory).
    """
    # Get the absolute path of the directory containing this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the project root (adjust the number of os.path.dirname calls if needed)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    # Construct the full path to the config.json file located at the project root.
    config_path = os.path.join(project_root, 'config', 'parameters.json')
    
    # Open and load the JSON configuration file.
    with open(config_path, 'r') as config_file:
        return json.load(config_file)