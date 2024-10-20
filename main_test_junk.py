from models.ensemble import stacking_model, load_data_for_model

def main():
    # Load the data
    X, y = load_data_for_model()

    # Run the ensemble models
    stacking_model(X, y)

if __name__ == "__main__":
    main()
