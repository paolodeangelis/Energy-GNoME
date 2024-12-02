import os
import pandas as pd
import yaml
from pathlib import Path
import typer

def find_project_root(start_path: str) -> str:
    """
    Recursively find the project root by looking for a data folder.
    
    Parameters:
    -----------
    start_path : str
        Starting path to search from
    
    Returns:
    --------
    str
        Path to the project root
    """
    current_path = start_path
    while current_path != os.path.dirname(current_path):
        # Check for data folder structure
        if all(os.path.exists(os.path.join(current_path, folder)) 
               for folder in ['data', 'data/raw', 'data/interim', 'data/final']):
            return current_path
        current_path = os.path.dirname(current_path)
    
    raise typer.BadParameter("Could not find project root with data folder structure")

def process_data_with_yaml(
    input_path: str, 
    yaml_config_path: str,
    keep_only_mapped: bool = True
) -> pd.DataFrame:
    """
    Process JSON data based on a YAML configuration file.
    
    Parameters:
    -----------
    input_path : str
        Path to the input JSON file in interim directory
    yaml_config_path : str
        Path to the YAML configuration file
    keep_only_mapped : bool, optional
        If True, keep only columns specified in the YAML mapping
        If False, rename specified columns but keep all others
    
    Returns:
    --------
    tuple: (processed DataFrame, final output path)
    """
    # Find project root by traversing up from the current script location
    current_path = os.path.abspath(os.path.curdir)
    project_root = find_project_root(current_path)

    
    # Read YAML configuration
    with open(yaml_config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    # Construct full paths
    full_input_path = os.path.join(project_root, input_path)
    final_output_path = os.path.join(project_root, config.get('path', ''))
    
    # Read JSON data
    df = pd.read_json(full_input_path)
    # print(df.columns)

    
    # Rename columns based on YAML configuration
    column_mapping = config.get('columns', {})
    df = df.rename(columns=column_mapping)
    # print(df.columns)
    
    
    if keep_only_mapped:
        # Use the new column names from the mapping
        df = df[list(column_mapping.values())]
        
    # Add note column if specified in config
    if 'note' in config:
        df['Note'] = config['note']

    return df, final_output_path

