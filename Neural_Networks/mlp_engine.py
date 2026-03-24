import numpy as np

class MLP_Network:
    
    def __init__(self, layer_sizes, activation_function, activation_derivative, 
                 last_layer_linear=True, init_method='uniform'):
        """
        layer_sizes: lista intów, np. [1, 5, 1] (wejście, ukryta, wyjście)
        activation_func: funkcja aktywacji, np. sigmoida, tanh
        """
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.last_layer_linear = last_layer_linear
        self.norm_params = {}
        
        # Inicjalizacja wag i biasów
        self.weights = []
        self.biases = []
        self._initialize_weights(init_method)
        
        # Pamięć prędkości dla Momentum
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        # Dla RMSProp
        self.s_w = [np.zeros_like(w) for w in self.weights]
        self.s_b = [np.zeros_like(b) for b in self.biases]
        
        # Listy do wizualizacji historii wag i straty
        self.weight_history = []
        self.loss_history = []

    def _initialize_weights(self, method):
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            
            if method == 'uniform':
                # Rozkład jednostajny [0, 1]
                w = np.random.uniform(0, 1, (n_in, n_out))
            elif method == 'xavier':
                # Inicjalizacja Xaviera
                limit = np.sqrt(6 / (n_in + n_out))
                w = np.random.uniform(-limit, limit, (n_in, n_out))
            elif method == 'he':
                # Inicjalizacja He
                limit = np.sqrt(6 / n_in)
                w = np.random.uniform(-limit, limit, (n_in, n_out))
            else:
                w = np.random.randn(n_in, n_out) * 0.1
                
            self.weights.append(w)
            self.biases.append(np.zeros((1, n_out)))

    def set_normalization(self, norm_dict):
        self.norm_params = norm_dict

    def _forward_train(self, X):
        """Forward pass zachowujący stany pośrednie dla backpropagation."""
        activations = [X]
        sums_weighted = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            sums_weighted.append(z)
            if i == len(self.weights) - 1 and self.last_layer_linear:
                a = z
            else:
                a = self.activation_function(z)
            activations.append(a)
        return activations, sums_weighted

    def train(self, X, Y, epochs, lr, batch_size=None, optimizer='sgd', momentum=0.9, beta=0.9, epsilon=1e-8, history_param=1000):
        """
        Główna pętla treningowa.
        batch_size=None -> Full Batch
        batch_size=1    -> Online (Stochastic)
        batch_size=k    -> Mini-batch 
        """
        n_samples = X.shape[0]
        if batch_size is None: batch_size = n_samples

        for epoch in range(epochs):
            # Mieszanie danych
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, n_samples, batch_size):
                x_batch = X_shuffled[i:i+batch_size]
                y_batch = Y_shuffled[i:i+batch_size]
                
                # Propagacja w przód i wstecz
                activations, sums = self._forward_train(x_batch)
                self._backward(activations, sums, y_batch, lr, optimizer, momentum, beta, epsilon)

            # Obliczenie i zapis błędu (znormalizowanego)
            y_pred_epoch = self.forward(X)

            # Zapisywanie wag do wizualizacji
            #self.weight_history.append([w.copy() for w in self.weights])

            if epoch % history_param == 0:
                epoch_loss = np.mean((Y - y_pred_epoch)**2)
                self.loss_history.append(epoch_loss)

    def _backward(self, activations, sums, Y, lr, optimizer='sgd', momentum=0.9, beta=0.9, epsilon=1e-8):
        """Implementacja wektorowego backpropagation."""
        m = Y.shape[0]
        num_layers = len(self.weights)
        
        # Błąd warstwy wyjściowej
        if self.last_layer_linear:
            delta = (activations[-1] - Y) / m
        else:
            delta = (activations[-1] - Y) * self.activation_derivative(sums[-1]) / m
            
        for i in reversed(range(num_layers)):
            # Obliczanie gradientów
            dw = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Propagacja błędu do warstwy niżej 
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(sums[i-1])
            
            # Aktualizacja wag
            if optimizer == 'momentum':
                # Uczenie gradientowe z momentem
                self.v_w[i] = momentum * self.v_w[i] + lr * dw
                self.v_b[i] = momentum * self.v_b[i] + lr * db
                self.weights[i] -= self.v_w[i]
                self.biases[i] -= self.v_b[i]
                
            elif optimizer == 'rmsprop':
                # Normalizacja gradientu RMSProp
                self.s_w[i] = beta * self.s_w[i] + (1 - beta) * (dw ** 2)
                self.s_b[i] = beta * self.s_b[i] + (1 - beta) * (db ** 2)
                self.weights[i] -= (lr / (np.sqrt(self.s_w[i]) + epsilon)) * dw
                self.biases[i] -= (lr / (np.sqrt(self.s_b[i]) + epsilon)) * db
                
            else:
                # Stochastic Gradient Descent (SGD)
                self.weights[i] -= lr * dw
                self.biases[i] -= lr * db

    def forward(self, X):
        """Standardowy forward pass dla predykcji."""
        
        if self.norm_params.get('czy_znormalizowac', False):
            x_min, x_max = self.norm_params['x_min'], self.norm_params['x_max']
            a = (X - x_min) / (x_max - x_min)
        else:
            a = X
            
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1 and self.last_layer_linear:
                a = z
            else:
                a = self.activation_function(z)
        
        if self.norm_params.get('czy_znormalizowac', False):
            y_min, y_max = self.norm_params['y_min'], self.norm_params['y_max']
            a = a * (y_max - y_min) + y_min
            
        return a