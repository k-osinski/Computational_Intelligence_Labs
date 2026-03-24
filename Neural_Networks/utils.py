import csv
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlp_engine import MLP_Network
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative


# Funkcja pomocnicza do treningu i ewaluacji sieci
def train_and_evaluate(variants_config, results_dict, X_train, y_train, X_test, y_test, 
                       layer_sizes, act_func, act_deriv, init_method='xavier', normalization_method='0_1', loss_hist=1000):
    """
    Funkcja trenująca modele z listy variants_config. 
    Jeśli model już istnieje w results_dict, zostanie on dotrenowany.
    normalization_method: '0_1' dla przedziału [0, 1] lub '-1_1' dla przedziału [-1, 1]
    """
    # Stałe normalizacji z danych treningowych
    x_min, x_max = X_train.min(), X_train.max()
    y_min, y_max = y_train.min(), y_train.max()

    if normalization_method == '-1_1':
        # Normalizacja do [-1, 1]
        X_train_norm = 2 * (X_train - x_min) / (x_max - x_min) - 1
        y_train_norm = 2 * (y_train - y_min) / (y_max - y_min) - 1
        X_test_norm = 2 * (X_test - x_min) / (x_max - x_min) - 1
    else:
        # Normalizacja danych do [0, 1]
        X_train_norm = (X_train - x_min) / (x_max - x_min)
        y_train_norm = (y_train - y_min) / (y_max - y_min)
        X_test_norm = (X_test - x_min) / (x_max - x_min)

    for v in variants_config:
        name = v['name']
        
        # Mechanizm dotrenowywania
        if name in results_dict and 'model' in results_dict[name]:
            mlp = results_dict[name]['model']
            previous_time = results_dict[name]['time']
            print(f"[*] DOTRENOWUJĘ istniejący model '{name}' o kolejne {v['epochs']} epok (lr={v['lr']})...")
        else:
            mlp = MLP_Network(layer_sizes, act_func, act_deriv, 
                              last_layer_linear=True, init_method=init_method)
            previous_time = 0.0
            print(f"[*] TWORZĘ NOWY model '{name}' i trenuję {v['epochs']} epok (lr={v['lr']})...")

        # Parametry optymalizatora
        opt = v.get('optimizer', 'sgd')
        mom = v.get('momentum', 0.9)
        beta_val = v.get('beta', 0.9)
        eps_val = v.get('epsilon', 1e-8)
        
        # Trening sieci
        start_time = time.time()
        mlp.train(X_train_norm, y_train_norm, epochs=v['epochs'], lr=v['lr'], 
                  batch_size=v['batch_size'], optimizer=opt, momentum=mom, 
                  beta=beta_val, epsilon=eps_val, history_param=loss_hist)
        current_time = time.time() - start_time
        total_time = previous_time + current_time
        
        # Predykcja i MSE na zbiorze testowym
        y_pred_test_norm = mlp.forward(X_test_norm)
        
        # Denormalizacja
        if normalization_method == '-1_1':
            y_pred_test = (y_pred_test_norm + 1) / 2 * (y_max - y_min) + y_min
        else:  
            y_pred_test = y_pred_test_norm * (y_max - y_min) + y_min

        mse_test = np.mean((y_test - y_pred_test)**2)
        
        # Zapisanie/Aktualizacja wyników w słowniku
        results_dict[name] = {
            'model': mlp, 
            'time': total_time,
            'mse': mse_test, 
            'history': mlp.weight_history
        }
        
        print(f"    -> Całkowity czas ukańczania: {total_time:6.2f}s | TEST MSE: {mse_test:8.4f}\n")
        
    return results_dict


# Wczytanie danych
def load_dataset(dataset_id):
    if dataset_id == 0:
        data = csv.reader(open('../mio1/regression/square-simple-training.csv'))
        test_data = csv.reader(open('../mio1/regression/square-simple-test.csv'))
    elif dataset_id == 1:
        data = csv.reader(open('../mio1/regression/steps-large-training.csv'))
        test_data = csv.reader(open('../mio1/regression/steps-large-test.csv'))
    elif dataset_id == 2:
        data = csv.reader(open('../mio1/regression/steps-small-training.csv'))
        test_data = csv.reader(open('../mio1/regression/steps-small-test.csv'))
    elif dataset_id == 3:
        data = csv.reader(open('../mio1/regression/multimodal-large-training.csv'))
        test_data = csv.reader(open('../mio1/regression/multimodal-large-test.csv'))
    elif dataset_id == 4:
        data = csv.reader(open('../mio1/regression/square-large-training.csv'))
        test_data = csv.reader(open('../mio1/regression/square-large-test.csv'))
    else:
        assert False, f"Wybrana baza danych nie jest dostępna."

    # --- Przygotowanie danych uczących ---
    x_list = []
    y_list = []
    for i, row in enumerate(data):
        if i > 0:
            if dataset_id ==3:
                x, y = row
            else:
                _, x, y = row
            x_list.append(float(x))
            y_list.append(float(y))
            
    # --- Przygotowanie danych testowych ---
    test_x_list = []
    test_y_list = []
    for i, row in enumerate(test_data):
        if i > 0:
            if dataset_id ==3:
                x, y = row
            else:
                _, x, y = row
            test_x_list.append(float(x))
            test_y_list.append(float(y))

    X_train = np.array(x_list).reshape(-1, 1)
    Y_train = np.array(y_list).reshape(-1, 1)
    X_test = np.array(test_x_list).reshape(-1, 1)
    Y_test = np.array(test_y_list).reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test


# Zapis modeli
def save_results(results_dict, filename="models.pkl"):
    """Zapisuje słownik z modelami do pliku."""
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Wyniki pomyślnie zapisane do pliku: {filename}")

def load_results(filename="models.pkl"):
    """Wczytuje słownik z modelami z pliku."""
    try:
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)
        print(f"Wyniki pomyślnie wczytane z pliku: {filename}")
        return results_dict
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {filename}. Zwracam pusty słownik.")
        return {}


# Wizualizacje
def plot_loss_curves(results, name=None):
    """
    Rysuje krzywą uczenia (Loss vs Epoki).
    Jeśli name jest podane, rysuje tylko jeden wariant.
    """
    plt.figure(figsize=(10, 6))
    
    models_to_plot = {name: results[name]} if name else results
        
    for k, v in models_to_plot.items():
        # Używamy skali logarytmicznej na osi Y, jeśli błędy bardzo mocno spadają
        plt.plot(v['model'].loss_history, label=k, linewidth=1.5)
        
    plt.title("Krzywa uczenia (MSE na zbiorze treningowym w czasie)")
    plt.xlabel("Epoka")
    plt.ylabel("MSE (znormalizowane)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(results, X_test, y_test, x_min, x_max, y_min, y_max, name=None, normalization_method='0_1'):
    """
    Nakłada predykcje modelu na prawdziwe dane testowe.
    Sortuje wartości X, aby wykres linii był czytelny (a nie był "bazgrołem").
    """
    # Sortowanie danych wzdłuż osi X
    sort_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sort_idx]
    y_sorted = y_test[sort_idx]
    
    # Przygotowanie znormalizowanych danych testowych
    if normalization_method == '-1_1':
        X_norm = 2 * (X_sorted - x_min) / (x_max - x_min) - 1
    else:
        X_norm = (X_sorted - x_min) / (x_max - x_min)

    # Predykcja
    model = results[name]['model']
    y_pred_norm = model.forward(X_norm)
    
    # Denormalizacja
    if normalization_method == '-1_1':
        y_pred = (y_pred_norm + 1) / 2 * (y_max - y_min) + y_min
    else:
        y_pred = y_pred_norm * (y_max - y_min) + y_min
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_sorted, y_sorted, color='black', alpha=0.5, label='Prawdziwe dane (Test)')
    plt.plot(X_sorted, y_pred, linewidth=3, label=f'Predykcja: {name}')
    plt.title(f'Wizualizacja dopasowania: {name} do danych testowych')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_mean_weights(results):
    plt.figure(figsize=(12, 5))

    for name, res in results.items():
        # Średnia wartość wag pierwszej warstwy w kolejnych epokach
        weight_means = [np.mean(h[0]) for h in res['history']]
        plt.plot(weight_means, label=name)
    
    plt.title("Wizualizacja zmian średnich wag (Warstwa 1)")
    plt.xlabel("Epoka")
    plt.ylabel("Średnia wartość wag")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_specific_weights(results, name=None):
    plt.figure(figsize=(12, 5))
    history = results['name']['history']
    
    # 3 pierwsze wagi łączące wejście z pierwszą warstwą ukrytą
    w1 = [h[0][0, 0] for h in history]
    w2 = [h[0][0, 1] for h in history]
    w3 = [h[0][0, 2] for h in history]
    
    plt.plot(w1, label='Waga 1 (Neuron 1)')
    plt.plot(w2, label='Waga 2 (Neuron 2)')
    plt.plot(w3, label='Waga 3 (Neuron 3)')
    
    plt.title("Trajektorie pojedynczych wag w warstwie 1 (Mini-batch)")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość wagi")
    plt.legend()
    plt.grid(True)
    plt.show()