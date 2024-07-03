# main.py

import data
import model
import train
import test

def main():
    # Load and preprocess the data
    train_data, val_data = data.load_data()
    
    # Build the model
    model_architecture = model.build_model()
    
    # Train the model
    train.train_model(model_architecture, train_data, val_data)
    
    # Test the model
    test.test_model(model_architecture, val_data)

if __name__ == "__main__":
    main()
