from pykalman import KalmanFilter
import numpy as np

# Definir el estado inicial y la matriz de transición
x_init = np.array([103, 145])  # Estado inicial (posición x, posición y)
transition_matrix = [[1, 0], [0, 1]]  # Matriz de transición para mantener el estado

# Definir la matriz de medición y el ruido de la medición
measurement_matrix = [[1, 0], [0, 1]]  # Medimos tanto la posición x como la posición y
measurement_noise = 0.1  # Ruido de la medición

# Definir el ruido del proceso
process_noise = np.array([[0.1, 0], [0, 0.1]])  # No hay cambio en la velocidad, solo en la posición

# Crear el filtro de Kalman
kf = KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=measurement_matrix,
                  initial_state_mean=x_init,
                  observation_covariance=measurement_noise,
                  transition_covariance=process_noise)

# Ejecutar el filtro de Kalman sin mediciones (predicción)
next_state_mean, next_state_covariance = kf.filter_update(x_init, kf.transition_covariance)

# Imprimir la predicción del próximo estado
print("Predicción para las próximas muestras:")
print("Próxima posición x:", next_state_mean[0])
print("Próxima posición y:", next_state_mean[1])
