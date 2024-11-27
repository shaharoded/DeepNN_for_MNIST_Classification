import torch
import numpy as np


class DeepNeuralNetwork:
    """
    A Deep Neural Network for classification tasks, implemented with PyTorch.
    Includes methods for training, prediction, and utility operations.
    The class will handle optimizations on the training process, early stops and allows for 
    batch normalization and L2 regularization.
    
    The class also monitors the training process after splitting X to train and validation, allowing 
    for progress logs in the terminal.
    """

    def __init__(
        self, layers_dims, num_iterations=100000, log_interval=100, learning_rate=0.009, batch_size=64,
        use_batchnorm=False, lambd=0.0, validation_split=0.2, early_stopping=True,
        stopping_steps=100, verbose = True, device=None
    ):
        """
        Initializes the neural network and training parameters.

        Parameters:
            layers_dims (list): List of layer dimensions (input to output).
            num_iterations (int): Maximum number of training steps (default: 1000).
            log_interval (int): Number of iterations between log statuses, if verbose = True (default: 100).
            earning_rate (float): Learning rate for gradient descent (default: 0.01).
            batch_size (int): Number of examples per batch (default: 64).
            use_batchnorm (bool): Whether to use batch normalization (default: False).
            lambd (float): Regularization parameter for L2 regularization (default: 0.0 = no regularization).
            validation_split (float): Fraction of data to use for validation (default: 0.2).
            early_stopping (bool): Whether to use early stopping (default: True).
            stopping_steps (int): Number of steps without improvement to stop training (default: 100).
            verbose (bool): Controls the progress prints in the training function (default: True).
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.layers_dims = layers_dims
        self.num_iterations = num_iterations
        self.log_interval = log_interval
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_batchnorm = use_batchnorm
        self.lambd = lambd
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.stopping_steps = stopping_steps
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = self.__initialize_parameters()
        self.verbose and print(f"\nUsing {self.device} device for computation.", flush=True)


    def __initialize_parameters(self):
        """
        Initializes parameters for an L-layer neural network using He initialization.
        """
        torch.manual_seed(42)
        parameters = {}
        L = len(self.layers_dims)

        for l in range(1, L):
            parameters[f"W{l}"] = torch.randn(
                self.layers_dims[l], self.layers_dims[l - 1], device=self.device
            ) * torch.sqrt(torch.tensor(2.0 / self.layers_dims[l - 1], device=self.device))
            parameters[f"b{l}"] = torch.zeros(self.layers_dims[l], 1, device=self.device)

        return parameters

    def __linear_forward(self, A, W, b):
        """
        Implements the linear part of forward propagation for a single layer.
        """
        Z = torch.mm(W, A) + b
        return Z, (A, W, b)


    def __activation_forward(self, A_prev, W, b, activation):
        """
        Implements forward propagation for LINEAR->ACTIVATION layer.
        Args:
            A_prev (torch.Tensor): Activations from the previous layer 
            W (torch.Tensor): Weight matrix of the current layer 
            b (torch.Tensor): Bias vector of the current layer 
            activation (str): Activation function to apply. Must be one of:
                              - "relu" for ReLU activation
                              - "softmax" for Softmax activation 
        """
        Z, linear_cache = self.__linear_forward(A_prev, W, b)
        if activation == "relu":
            A = torch.maximum(Z, torch.tensor(0.0, device=self.device))
        elif activation == "softmax":
            Z_exp = torch.exp(Z - torch.max(Z, dim=0, keepdim=True).values)
            A = Z_exp / torch.sum(Z_exp, dim=0, keepdim=True)
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'softmax'.")
        return A, {"linear_cache": linear_cache, "activation_cache": Z}

    
    def __apply_batchnorm(self, A):
        """
        Perform batch normalization on the activation values of a given layer (normalizing each layer of the model).

        :param A: activation values of a given layer (tensor of shape [features, batch_size])
        :param epsilon: small constant to avoid division by zero

        :return: NA – the normalized activation values
        """
        epsilon = 1e-8
        # Compute mean and variance for each feature
        mean = A.mean(dim=1, keepdim=True)
        var = A.var(dim=1, keepdim=True, unbiased=False)
        
        # Return normalize activations
        return (A - mean) / torch.sqrt(var + epsilon)


    def forward_propagation(self, X):
        """
        Implements forward propagation through the entire network.
        Computation performed: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation.
        Args:
            X: input data, torch tensor of shape (input size, number of examples)
        
        Returns:
            AL – the last post-activation value
            caches – a list of all the cache objects generated by the linear_forward function
        """
        caches = []
        A = X.to(self.device)
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            W, b = self.parameters[f"W{l}"], self.parameters[f"b{l}"]
            A, cache = self.__activation_forward(A_prev, W, b, activation="relu")

            if self.use_batchnorm:
                A = self.__apply_batchnorm(A)
            caches.append(cache)

        W, b = self.parameters[f"W{L}"], self.parameters[f"b{L}"]
        AL, cache = self.__activation_forward(A, W, b, activation="softmax")
        caches.append(cache)
        return AL, caches


    def compute_cost(self, AL, Y):
        """
        Computes the cross-entropy cost with an option for L2 regularization.
        Pass lambd=0 to disable regularization.

        Parameters:
            AL (torch.Tensor): The predicted probabilities, shape (num_classes, num_examples).
            Y (torch.Tensor): The true labels, shape (num_classes, num_examples).
            parameters (dict): Dictionary of the neural network parameters.
            lambd (float): Regularization parameter (default: 0.0 in the main function).

        Returns:
            cost (torch.Tensor): Regularized cost.
        """
        N = Y.shape[1]
        cross_entropy_cost = -torch.sum(Y * torch.log(AL + 1e-8)) / N
        L2_term = (self.lambd / (2 * N)) * sum(
            torch.sum(torch.square(self.parameters[f"W{l}"])) for l in range(1, len(self.parameters) // 2 + 1)
        )
        return cross_entropy_cost + L2_term


    def __linear_backward(self, dZ, cache):
        """
        Implements the linear part of the backward propagation.
        """
        A_prev, W, _ = cache
        m = A_prev.shape[1]
        dW = torch.mm(dZ, A_prev.T) / m + (self.lambd / m) * W
        db = torch.sum(dZ, dim=1, keepdim=True) / m
        dA_prev = torch.mm(W.T, dZ)
        return dA_prev, dW, db
    
    
    def __linear_activation_backward(self, dA, cache, activation):
        """
        Implements the backward propagation for a LINEAR->ACTIVATION layer.
        """
        linear_cache = cache["linear_cache"]
        activation_cache = cache["activation_cache"]

        if activation == "relu":
            dZ = dA * (activation_cache > 0).float().to(self.device)
        elif activation == "softmax":
            dZ = dA
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'softmax'.")

        dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        return dA_prev, dW, db


    def backward_propagation(self, AL, Y, caches):
        """
        Implements the backward propagation for the entire network.
        """
        grads = {}
        L = len(caches)
        Y = Y.to(self.device)
        dAL = AL - Y

        current_cache = caches[-1]
        grads[f"dA{L - 1}"], grads[f"dW{L}"], grads[f"db{L}"] = self.__linear_activation_backward(
            dAL, current_cache, "softmax"
        )

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            grads[f"dA{l}"], grads[f"dW{l + 1}"], grads[f"db{l + 1}"] = self.__linear_activation_backward(
                grads[f"dA{l + 1}"], current_cache, "relu"
            )

        return grads


    def update_parameters(self, grads):
        """
        Updates parameters using gradient descent.
        """
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]
            
            
    def __split_data(self, X, Y, validation_split):
        """
        Splits the dataset into training and validation sets.
        """
        assert X.shape[1] == Y.shape[1], "Mismatch between X and Y sample counts."
        np.random.seed(42)
        N = X.shape[1]
        permutation = np.random.permutation(N)
        X_shuffled, Y_shuffled = X[:, permutation], Y[:, permutation]
        num_val = int(N * validation_split)
        return X_shuffled[:, num_val:], X_shuffled[:, :num_val], Y_shuffled[:, num_val:], Y_shuffled[:, :num_val]


    def __to_tensor(self, data, dtype=torch.float32):
        """
        Converts data to PyTorch tensors with the correct dtype and device.
        """
        if isinstance(data, torch.Tensor):
            return data.clone().detach().to(dtype=dtype, device=self.device)
        return torch.tensor(data, dtype=dtype, device=self.device)


    def __compute_warm_up_iterations(self, N, L, k=3, reference_depth=4):
        """
        Computes the warm-up steps for early stopping based on network depth.
        """
        iterations_per_epoch = (N + self.batch_size - 1) // self.batch_size
        depth_factor = L / reference_depth
        return max(int(k * iterations_per_epoch * depth_factor), iterations_per_epoch)
        
    
    def train(self, X, Y):
        """
        Trains the neural network using minibatch stochastic gradient descent with early stopping.

        Includes logging of training and validation costs, warm-up iterations, and validation cost tracking.

        Args:
            X (numpy.ndarray): Input data of shape (input_size, number_of_examples).
            Y (numpy.ndarray): True labels of shape (num_classes, number_of_examples).

        Returns:
            costs (list): List of tuples (training_cost, validation_cost) logged every `log_interval` steps.
            best_parameters (dict): Best parameters based on validation cost.
        """
        X_train, X_val, Y_train, Y_val = self.__split_data(X, Y, self.validation_split)
        X_train, Y_train = self.__to_tensor(X_train), self.__to_tensor(Y_train)
        X_val, Y_val = self.__to_tensor(X_val), self.__to_tensor(Y_val)

        N = X_train.shape[1]  # Number of training examples
        L = len(self.layers_dims) - 1  # Number of layers (excluding input)
        iterations_per_epoch = (N + self.batch_size - 1) // self.batch_size
        warm_up_steps = self.__compute_warm_up_iterations(N, L)

        if self.verbose:
            print(
                "\n-------- Model Initialization --------\n"
                f"Model initialized with {L} layers.\n"
                f"Early stopping patience: {self.stopping_steps} steps.\n"
                f"Minimum warm-up iterations: {warm_up_steps}.\n"
                f"Iterations per epoch: {iterations_per_epoch}.\n"
                "--------------------------------------\n",
                flush=True
            )

        best_val_cost = float("inf")
        step = 0  # Initialize step counter
        best_step = 0
        patience_counter = 0
        best_parameters = None
        costs = []
        train_costs_batch = []
        val_costs_batch = []

        # Loop until exceeded the maximum number of steps
        while step < self.num_iterations:
            # Shuffle the training data
            permutation = torch.randperm(N)
            X_train_shuffled, Y_train_shuffled = X_train[:, permutation], Y_train[:, permutation]

            for i in range(0, N, self.batch_size):                
                
                # Create mini-batches
                X_batch = X_train_shuffled[:, i:i + self.batch_size]
                Y_batch = Y_train_shuffled[:, i:i + self.batch_size]

                # Forward propagation
                AL, caches = self.forward_propagation(X_batch)
                train_cost = self.compute_cost(AL, Y_batch)

                # Backward propagation
                grads = self.backward_propagation(AL, Y_batch, caches)
                self.update_parameters(grads)

                # Compute validation cost
                AL_val, _ = self.forward_propagation(X_val)
                val_cost = self.compute_cost(AL_val, Y_val).item()

                # Track costs for averaging
                train_costs_batch.append(train_cost.item())
                val_costs_batch.append(val_cost)
                
                # Track validation cost and save best parameters
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    best_parameters = self.parameters.copy()
                    best_step = step
                    patience_counter = 0
                elif step > warm_up_steps:
                    patience_counter += 1

                # Early stopping condition
                if self.early_stopping and patience_counter >= self.stopping_steps:
                    completed_epochs = step // iterations_per_epoch
                    if self.verbose:
                        print(
                        f"Early stopping at step {step} ({completed_epochs} epochs completed). Best Validation Cost: {best_val_cost:.4f}, Achieved at Step {best_step}", flush=True
                        )
                    # Calculate accuracy
                    train_accuracy = self.predict(X_train, Y_train)
                    val_accuracy = self.predict(X_val, Y_val)
                    if self.verbose:
                        print(f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%", flush=True)
                    return costs, best_parameters, train_accuracy, val_accuracy


                # Log averaged costs every `log_interval` steps
                if step % self.log_interval == 0 and step > 0:
                    avg_train_cost = sum(train_costs_batch) / len(train_costs_batch)
                    avg_val_cost = sum(val_costs_batch) / len(val_costs_batch)
                    costs.append((avg_train_cost, avg_val_cost))
                    train_costs_batch.clear()
                    val_costs_batch.clear()
                    if self.verbose:
                        print(
                        f"Step {step}: Avg Training Cost = {avg_train_cost:.4f}, Avg Validation Cost = {avg_val_cost:.4f}.", flush=True
                        )
                
                # Increment step counter
                step += 1

        completed_epochs = self.num_iterations // iterations_per_epoch
        if self.verbose:
            print(
            f"Training completed after {self.num_iterations} steps ({completed_epochs} epochs).\n"
            f"Best Validation Cost: {best_val_cost:.4f}, Achieved at Step {best_step}",
            flush=True
            )
        # Calculate accuracy
        train_accuracy = self.predict(X_train, Y_train)
        val_accuracy = self.predict(X_val, Y_val)
        if self.verbose:
            print(f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%", flush=True)

        return costs, best_parameters, train_accuracy, val_accuracy


    def predict(self, X, Y):
        """
        Calculates the accuracy of the trained neural network on the given data.

        Parameters:
            X (numpy.ndarray): Input data of shape (height*width, number_of_examples).
            Y (numpy.ndarray): True labels of shape (num_of_classes, number_of_examples).

        Returns:
            accuracy (float): Accuracy of the neural network as a percentage.
        """
        X, Y = self.__to_tensor(X), self.__to_tensor(Y)
        AL, _ = self.forward_propagation(X)
        predictions = torch.argmax(AL, dim=0)
        true_labels = torch.argmax(Y, dim=0)
        return (predictions == true_labels).float().mean().item() * 100