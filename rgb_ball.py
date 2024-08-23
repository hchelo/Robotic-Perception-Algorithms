import cv2
import numpy as np

# Iniciar la captura de video
cap = cv2.VideoCapture('futbol.mp4')

# Inicializar el contador de frames
frame_count = 0

while True:
    # Leer el siguiente cuadro del video
    ret, frame = cap.read()
    
    if not ret:
        break  # Si no hay más cuadros, salir del bucle

    # Incrementar el contador de frames
    frame_count += 1

    # Obtener la altura y ancho del cuadro
    height, width, _ = frame.shape
    
    # Crear una máscara del mismo tamaño que el cuadro original
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Recorrer todos los píxeles del cuadro y cambiar su color
    for y in range(50, height):
        for x in range(width):
            pixel = frame[y, x]
            if pixel[2] <= 190 and pixel[2] >= 100 and pixel[1] <= 201 and pixel[1] >= 158 and pixel[0] <= 82 and pixel[0] >= 34:
                mask[y, x] = 255  # Establecer píxeles correspondientes al blob como blanco
    
    # Aplicar operaciones morfológicas (cierre y dilatación) a la máscara
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Encontrar el contorno del blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar un círculo alrededor del contorno del blob en el cuadro original
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
    
    # Mostrar el cuadro resultante
    cv2.imshow('Video', frame)

    # Detener la ejecución del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
