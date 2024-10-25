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

def display_head_and_tail(data, rows=5):
    """
    Display the head and tail of each DataFrame stored in the pickle file.
    """
    if data is not None:
        for key, value in data.items():
            print(f"\nKey: {key}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"  Subkey: {sub_key}")
                    if isinstance(sub_value, pd.DataFrame):
                        print(f"  Head of {sub_key}:\n{sub_value.head(rows)}")
                        print(f"  Tail of {sub_key}:\n{sub_value.tail(rows)}")
                    else:
                        print(f"  {type(sub_value)} - Not a DataFrame")
            else:
                if isinstance(value, pd.DataFrame):
                    print(f"Head of {key}:\n{value.head(rows)}")
                    print(f"Tail of {key}:\n{value.tail(rows)}")
                else:
                    print(f"{type(value)} - Not a DataFrame")
    else:
        print("No data to display.")

if __name__ == '__main__':
    # Load and display the pickle file contents
    data = load_pickle_file(file_path)
    display_head_and_tail(data, rows=5)  # Modify 'rows' to display more or fewer rows
