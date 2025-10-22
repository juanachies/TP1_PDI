import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, xor_img=False, blocking=False, xorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if xor_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if xorbar:
        plt.xorbar()
    if new_fig:        
        plt.show(block=blocking)


img = cv2.imread('Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE) 

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
    pad_y = m // 2
    pad_x = n // 2
    img = cv2.copyMakeBorder(imagen, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(imagen, cmap='gray')
    axs[0].set_title('Original')
    
    img_ecualizada = np.empty_like(img)

    # Recorrer cada píxel y aplicar equalizeHist a la ventana y tomar el píxel central
    for y in range(pad_y, alto + pad_y):
        for x in range(pad_x, ancho + pad_x):
            kernel = img[y - pad_y : y - pad_y + m, x - pad_x : x - pad_x + n] 
            kernel_eq = cv2.equalizeHist(kernel)
            img_ecualizada[y, x] = kernel_eq[pad_y, pad_x] 
    
    img_ruido = img_ecualizada[pad_y : pad_y + alto, pad_x : pad_x + ancho]
    axs[1].imshow(img_ruido, cmap='gray')
    axs[1].set_title(f'Ecualización local {m}x{n}')

    output = cv2.medianBlur(img_ruido,3)
    axs[2].imshow(output, cmap='gray')
    axs[2].set_title('Filtrado de ruido')

    plt.show()


ecualizacion_hist(img, 20, 20)