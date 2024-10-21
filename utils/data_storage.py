import pickle
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, storage

# Directory to store pickle files
save_dir = 'local_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Firebase initialization
#cred = credentials.Certificate('firebase_credentials.json')  # Replace with your Firebase credentials file
#firebase_admin.initialize_app(cred, {
#    'storageBucket': 'your-firebase-project-id.appspot.com'  # Replace with your Firebase project ID
#})
#bucket = storage.bucket()

def append_data_to_pickle(new_data):
    """
    Append new data to the historical data stored in a pickle file.
    """
    file_path = os.path.join(save_dir, "historical_data.pickle")

    # Load the existing data if the pickle file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
        print(f"Loaded existing historical data from {file_path}")
    else:
        existing_data = {}
        print(f"No existing data found, creating a new file.")

    # Append new data
    for key, value in new_data.items():
        if key in existing_data:
            if isinstance(existing_data[key], pd.DataFrame):
                existing_data[key] = pd.concat([existing_data[key], value]).drop_duplicates()
            else:
                print(f"Warning: {key} is not a DataFrame.")
        else:
            existing_data[key] = value

    # Save the updated data back to the pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(existing_data, f)
    print(f"Appended new historical data and saved to {file_path}")

#def upload_to_firebase(local_file_path, remote_file_name):
#    """
#    Upload the pickle file to Firebase Storage.
#    """
#    blob = bucket.blob(remote_file_name)
#    blob.upload_from_filename(local_file_path)
#    print(f"{local_file_path} uploaded to {remote_file_name}")
