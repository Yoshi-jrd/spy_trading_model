import pickle

# Path to the pickle file
file_path = 'best_params.pkl'

# Define the new LSTM parameters
new_lstm_params = {
    'model__units': 100,
    'model__dropout': 0.2,
    'model__learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 50
}

# Load existing parameters
with open(file_path, 'rb') as file:
    best_params = pickle.load(file)

# Update LSTM parameters
best_params['LSTM'] = new_lstm_params

# Save the updated parameters back to the pickle file
with open(file_path, 'wb') as file:
    pickle.dump(best_params, file)

print("Updated best_params.pkl with new LSTM settings:")
print(best_params)
