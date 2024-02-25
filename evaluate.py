import numpy as np
import tensorflow as tf
import json
import uproot
import awkward as ak
import pandas as pd

from my_functions import selection_criteria
cuts = "(METcorrected_pt > 250) & ((nVetoElectrons + nLooseMuons) == 0) & (njets >= 3) & (nbjets >= 1)"

def apply_cuts_and_get_data(tree, cuts):
    return tree.arrays(tree.keys(), cuts=cuts)

# Function to read variable names, mean values, and standard deviations from CSV
def read_variable_norm(file_path):
    norm_data = pd.read_csv(file_path)
    return norm_data

# Function to normalize data using the provided mean and standard deviation
def normalize_data(data, norm_data):
    for variable in norm_data['Unnamed: 0']:
        mean = norm_data[norm_data['Unnamed: 0'] == variable]['mu'].values[0]
        std = norm_data[norm_data['Unnamed: 0'] == variable]['std'].values[0]
        data[variable] = (data[variable] - mean) / std
    return data

# Function to load the binary classification model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model

# Function to perform binary classification using the loaded model
def classify_data(model, data):
    # Perform binary classification
    predictions = model.predict(data)
    return predictions

# Function to read input branches from the root tree
def read_root_tree(input_file_path, tree_name, branches):
    root_file = uproot.open(input_file_path)
    root_tree = root_file[tree_name]
    data = root_tree.arrays(branches, cuts)
    return data


# Function to read input data from a Parquet file
def read_parquet_file(input_file_path, columns):
    data = pd.read_parquet(input_file_path, columns=columns)
    return data

def save_output_root_tree(output_file_path, tree_name, data):

    #print("Data type:", type(data))
    #print("Data:", data)

    # Convert the data to a NumPy structured array
    structured_data = np.empty(len(data), dtype=[(name, float) for name in data.fields])
    for field in data.fields:
        structured_data[field] = ak.to_numpy(data[field])

    df = pd.DataFrame(structured_data)

    file = uproot.recreate(output_file_path)
    file[tree_name] = df


# Main function
def main(input_file_path, output_file_path, config_file_path):
    # Load configuration from the JSON file
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Make sure config is a dictionary
    if not isinstance(config, dict):
        raise ValueError("The JSON configuration file should contain a dictionary.")

    # Extract configuration values
    model_path = config['model_path']
    input_type = config['input_type']  # 'root' or 'parquet'
    input_tree_name = config['input_tree_name']
    input_branches = config['input_branches']
    evaluation_branches = config['evaluation_branches']
    output_column_name = config['output_column_name']
    output_tree_name = config['output_tree_name']

    # Load the trained model
    model = load_model(model_path)

    # Read input data either from root tree or parquet file based on 'input_type'
    if input_type == 'root':
        data = read_root_tree(input_file_path, input_tree_name, input_branches)
    elif input_type == 'parquet':
        data = read_parquet_file(input_file_path, input_branches)
    else:
        raise ValueError("Invalid input_type. It should be either 'root' or 'parquet'.")

    # Prepare the data for evaluation using the specified evaluation branches
    x_data = np.column_stack([data[branch] for branch in evaluation_branches])

    # Convert awkward array to regular numpy array
    x_data_np = ak.to_numpy(x_data)

    # Load variable normalization data from CSV
    variable_norm_data = read_variable_norm('variable_norm.csv')

    # Create a DataFrame from the numpy array and apply normalization
    x_data_df = pd.DataFrame(x_data_np, columns=evaluation_branches)
    x_data_normalized_df = normalize_data(x_data_df, variable_norm_data)

    # Convert the normalized DataFrame back to a numpy array
    x_data_normalized_np = x_data_normalized_df.to_numpy()

    # Print shapes for troubleshooting
    print("x_data shape:", x_data_np.shape)
    print("Model input shape:", model.input_shape)
    #print(x_data_normalized_np)

    # Perform binary classification
    predictions = classify_data(model, x_data_normalized_np)
    print(predictions)

    # Add the classifier output to the data as a new branch
    data[output_column_name] = predictions.flatten()
    print(data[output_column_name])

    # Save the output root tree with the classifier output
    save_output_root_tree(output_file_path, output_tree_name, data)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python script.py input_file.root output_file.root config.json")
    else:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        config_file_path = sys.argv[3]
        main(input_file_path, output_file_path, config_file_path)
