import cv2 as cv
import numpy as np

# Iniciar la captura de video
cap = cv.VideoCapture('futbol.mp4')

while True:
    # Leer el siguiente cuadro del video
    ret, frame = cap.read()
    
    if not ret:
        break  # Si no hay más cuadros, salir del bucle

    # Convertir el cuadro a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Aplicar un desenfoque gaussiano para reducir el ruido
    gray_blur = cv.GaussianBlur(gray, (23, 23), 1.5)

    # Aplicar la transformada de Hough para detectar círculos
    circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, 1.3, minDist=50, param1=150, param2=60, minRadius=10, maxRadius=30)

    # Si se detectaron círculos, dibujarlos en el cuadro original
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for c in circles[0, 0:1]:
            cv.circle(frame, (c[0], c[1]), c[2], (0, 255, 0), 3)
            cv.circle(frame, (c[0], c[1]), 1, (0, 0, 255), 5)

    # Mostrar el cuadro resultante
    cv.imshow('Círculos detectados', frame)

    # Detener la ejecución del bucle si se presiona la tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv.destroyAllWindows()
