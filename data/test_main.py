from data_loader import load_volatility_data  # Import the function from data_loader

def main():
    # Load volatility data
    volatility_data = load_volatility_data()
    
    # Output the volatility metrics for review
    print("\nVolatility Data:")
    for key, value in volatility_data.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
