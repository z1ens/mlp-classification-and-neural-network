
# Bike vs. Plane Image Classifier

This README outlines the implementation details and usage instructions for a neural network-based image classifier designed to distinguish between bikes and planes. The model leverages a Multilayer Perceptron (MLP) architecture and is trained on a dataset split into training and validation subsets.

## Model Architecture

The neural network, encapsulated within the Network class, is constructed dynamically based on a provided architecture list that specifies the input, hidden, and output layer dimensions. The model employs Rectified Linear Unit (ReLU) activations for intermediate layers and a sigmoid activation function for the output layer to predict class probabilities in this binary classification task.

### Initialization
- **Weights**: Initialized randomly following a normal distribution.
- **Biases**: Initialized to zero. 

### Activation Functions
- **ReLU**: for intermediate layers, with implemented forward and backward propagation functions.
- **Sigmoid**: for the output layer, also with forward and backward propagation functions.

### Forward and Backward Propagation
- **Forward Propagation**: Implemented through layer_forward and forward functions, iteratively calculating layer outputs and activations.
- **Backward Propagation**: Utilizes layer_backward and back_propagation functions for calculating and propagating error derivatives, updating gradients for weights and biases.

### Evaluation
Methods are provided to evaluate the prediction accuracy of the neural network on the validation set. Users are encouraged to experiment with different network and training parameters to optimize predictive performance.

### Getting Started

1. Ensure you have the necessary Python environment and dependencies installed.
2. Clone this repository to your local machine.
3. Load your dataset in the appropriate format as mentioned in the dataset preparation guidelines
4. Run main.py to train the model with your dataset.
5. Adjust parameters in main.py as necessary to improve performance.

## License

[MIT](https://choosealicense.com/licenses/mit/)


