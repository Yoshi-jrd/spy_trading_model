import pickle
import os
import pandas as pd

# Path to the pickle file
file_path = os.path.join('local_data', 'historical_data.pickle')

def load_pickle_file(file_path):
    """
    Load and display the contents of the pickle file.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        print(f"No file found at {file_path}")
        return None

def display_data(data, rows=5):
    """
    Display the actual data from the pickle file, showing the first few rows.
    """
    if data is not None:
        for key, value in data.items():
            print(f"\nKey: {key}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"  Subkey: {sub_key}")
                    if isinstance(sub_value, pd.DataFrame) or isinstance(sub_value, pd.Series):
                        print(sub_value.head(rows))  # Print the first few rows of the DataFrame/Series
                    else:
                        print(f"  {type(sub_value)} - Not a DataFrame/Series")
            else:
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    print(value.head(rows))  # Print the first few rows of the DataFrame/Series
                else:
                    print(f"{type(value)} - Not a DataFrame/Series")
    else:
        print("No data to display.")

if __name__ == '__main__':
    # Load and display the pickle file contents
    data = load_pickle_file(file_path)
    display_data(data, rows=5)  # Modify 'rows' to display more or fewer rows
