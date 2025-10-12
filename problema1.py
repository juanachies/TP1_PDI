import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


img = cv2.imread('Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE) 
imshow(img)

# Histograma de la imagen
hist, bins = np.histogram(img.flatten(), 256, [0, 256]) 
hist2 = cv2.calcHist([img], [0], None, [256], [0, 256]) 
max(abs(hist.flatten() - hist2.flatten()))
img_heq = cv2.equalizeHist(img)
plt.plot(bins[:-1], hist)
plt.hist(img.flatten(), 256, [0, 256])
plt.title('Histograma')
plt.show()



def ecualizacion_hist(imagen: cv2, m: int, n: int):
    '''
    Función que realiza ecualización local del histograma
    '''
    ancho, alto = imagen.shape

    # Calculamos cuánto hay que rellenar
    pad_y = n // 2
    pad_x = m // 2
    img = cv2.copyMakeBorder(imagen, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    imshow(img)
    
    img_ecualizada = np.empty_like(img)

    # Recorrer cada píxel y aplicar equalizeHist a la ventana y tomar el píxel central
    for i in range(ancho):
        for j in range(alto):
            kernel = img[i : i + n, j : j + m]   
            kernel_eq = cv2.equalizeHist(kernel)
            img_ecualizada[i, j] = kernel_eq[pad_y, pad_x]

    imshow(img_ecualizada, title=f"Ecualización local {m}x{n}")


ecualizacion_hist(img, 15, 15)