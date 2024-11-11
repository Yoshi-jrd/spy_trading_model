import pickle

# Replace with the actual path to your pickle file
file_path = 'best_params.pkl'

# Load and print the contents of the pickle file
with open(file_path, 'rb') as file:
    best_params = pickle.load(file)
    print("Contents of best_params.pkl:")
    print(best_params)
