import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('futbol2.jpg')
# Obtener la altura y ancho de la imagen
height, width, _ = img.shape
#cv2.imshow('Imagen 1', img)
# Definir el nuevo color en formato RGB
            # b g r
new_color = (0, 255, 0) # rojo

# Recorrer todos los píxeles de la imagen y cambiar su color
for y in range(height):
    for x in range(width):
        pixel = img[y, x]
        #print("Pixel ({}, {}) = ({}, {}, {})".format(x, y, pixel[2], pixel[1], pixel[0]))
        b, g, r = pixel[0], pixel[1], pixel[2]
        #if 34 <= r <= 82 and 158 <= g <= 201 and 100 <= b <= 190:
        if (r > 95) and (g > 40) and (b > 20) and (max(r,g,b)-min(r,g,b)>15) and (abs(r-g)>15) and (r>g) and (r>b):
            img[y, x] = new_color;
        #else:
        #    img[y, x] = (0, 0, 0);   

# Mostrar la imagen resultante
cv2.imshow('Imagen 2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()