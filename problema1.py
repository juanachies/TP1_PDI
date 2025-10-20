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
def histograma(img): 
    hist, bins = np.histogram(img.flatten(), 256, [0, 256]) 
    hist2 = cv2.calcHist([img], [0], None, [256], [0, 256]) 
    max(abs(hist.flatten() - hist2.flatten()))
    img_heq = cv2.equalizeHist(img)
    plt.figure()
    plt.plot(bins[:-1], hist)
    plt.hist(img.flatten(), 256, [0, 256])
    plt.title('Histograma')
    plt.show()

histograma(img)


def ecualizacion_hist(imagen: cv2, m: int, n: int):
    '''
    Función que realiza ecualización local del histograma
    '''
    alto, ancho = imagen.shape

    # Calculamos cuánto hay que rellenar
    pad_y = n // 2
    pad_x = m // 2
    img = cv2.copyMakeBorder(imagen, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(imagen, cmap='gray')
    axs[0].set_title('Original')
    
    img_ecualizada = np.empty_like(img)

    # Recorrer cada píxel y aplicar equalizeHist a la ventana y tomar el píxel central
    for x in range(pad_x, alto + pad_x):
        for y in range(pad_y, ancho + pad_y):
            kernel = img[x - pad_x : x - pad_x + m, y - pad_y : y - pad_y + n] 
            kernel_eq = cv2.equalizeHist(kernel)
            img_ecualizada[x, y] = kernel_eq[pad_y, pad_x]
    
    output = img_ecualizada[pad_x : pad_x + alto, pad_y : pad_y + ancho]
    axs[1].imshow(output, cmap='gray')
    axs[1].set_title(f'Ecualización local {m}x{n}')

    plt.show()


ecualizacion_hist(img, 20, 20)