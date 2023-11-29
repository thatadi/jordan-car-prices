# jordan-car-prices
# Car Prices Prediction with Feedforward Neural Network

This repository contains code for predicting car prices using a feedforward neural network. The neural network is built using TensorFlow and trained on the Jordan car dataset, leveraging various attributes of car models to predict their prices.

## Dataset
The 'Jordan car dataset' contains information about different car models along with their attributes and prices. The dataset includes features such as mileage, year, make, model, etc., with the target variable being the price of the car.

## Neural Network Architecture
The neural network is constructed using TensorFlow's Keras API. Here is the general architecture of the neural network used for this prediction task:

- **Input Layer**: The input layer accepts the attributes of the car models.
- **Hidden Layers**: The network comprises multiple dense (fully connected) hidden layers with ReLU activation functions to capture complex relationships in the data.
- **Output Layer**: The output layer produces a single value representing the predicted price. It uses a linear activation function since this is a regression task.

## Files Included
- `jordan_car.ipynb`: Jupyter notebook containing the Python code for building and training the neural network.
- `car_prices_jordan.csv`: Sample dataset used for training and testing the model (replace with your dataset).

## Instructions
1. **Dataset Preparation**: Replace the `dataset.csv` file with your car dataset.
2. **Dependencies**: Ensure you have the necessary libraries installed, such as TensorFlow, NumPy, Pandas, etc.
3. **Training the Model**: Run the `car_prices_prediction.ipynb` notebook to train the neural network on your dataset. Adjust hyperparameters and model architecture as needed.
4. **Evaluation**: Evaluate the model's performance on test data and assess metrics like Mean Squared Error (MSE), R2 Score , RMSE to gauge the model's accuracy.
5. **Predictions**: Use the trained model to make predictions on new or unseen data to estimate car prices.

## Acknowledgments
- The Jordan car dataset used in this project (replace with your actual data source).
- References to online resources and documentation related to TensorFlow and neural network regression.

## Notes
- Experiment with different architectures, hyperparameters, and preprocessing techniques to optimize the model's performance.
- This repository serves as a starting point for predicting car prices using neural networks and can be extended for further analysis or improvements.

Feel free to explore, modify, and utilize this codebase for your car price prediction tasks. For any questions or suggestions, please feel free to open an issue or contact me.

