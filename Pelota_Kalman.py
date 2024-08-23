import cv2
import numpy as np
from pykalman import KalmanFilter

# Definir la matriz de covarianza de transición
transition_covariance = np.eye(2) * 0.1  # Varianzas iguales en ambas dimensiones

# Crear el filtro de Kalman con la matriz de covarianza de transición definida
kf = KalmanFilter(initial_state_mean=[0, 0], transition_covariance=transition_covariance)

def detecta_pelota(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  

        height, width, _ = frame.shape
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        red_mask = np.logical_and.reduce((frame[:, :, 2] <= 190, frame[:, :, 2] >= 100, 
                                        frame[:, :, 1] <= 201, frame[:, :, 1] >= 158, 
                                        frame[:, :, 0] <= 82, frame[:, :, 0] >= 34))

        mask[150:, :][red_mask[150:, :]] = 255

        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            center = np.array([x, y])  # Coordenadas del centro de la pelota
            
            # Dibujar la pelota detectada
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.putText(frame, "Pelota", (int(x-radius), int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Convertir center a una matriz numpy bidimensional
            center = center.reshape((1, 2))
            
            # Actualizar el filtro de Kalman con la nueva medición
            next_state_mean, next_state_covariance = kf.filter_update(center, kf.transition_covariance)
            
            # Obtener la predicción del próximo estado
            predicted_x, predicted_y = next_state_mean[0], next_state_mean[1]
            
            # Dibujar un círculo en la posición predicha
            cv2.circle(frame, (int(predicted_x), int(predicted_y)), int(radius), (0, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'futbol.mp4'
    detecta_pelota(video_path)
