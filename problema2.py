import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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


img = cv2.imread('formulario_01.png',cv2.IMREAD_GRAYSCALE) 
imshow(img)

# Umbralizamos imagen
th = 87
img_th = img>th #  matriz booleana
print(img_th.dtype)
print(np.unique(img_th))

img_th = np.uint8(img>th)*255
print(img_th.dtype)
print(np.unique(img_th))

imshow(img_th, title="Umbralado manual")

img_bool = img_th == 0




# --- Suma por filas (para detectar líneas horizontales) ---
row_sum = np.sum(img_bool, axis=1)
th_row = 0.5 * np.max(row_sum)
line_mask = row_sum > th_row  # True donde hay línea

# --- Encontrar los límites de cada renglón ---
renglones = []
en_linea = False
for i, val in enumerate(line_mask):
    if val and not en_linea:
        y1 = i
        en_linea = True
    elif not val and en_linea:
        y2 = i
        renglones.append((y1, y2))
        en_linea = False

# --- Generar subimágenes entre líneas ---
subimgs = []
for i in range(len(renglones)-1):
    y1 = renglones[i][1]
    y2 = renglones[i+1][0]
    subimgs.append(img[y1:y2, :])

# --- Mostrar todos los renglones en una sola figura ---
fig, axes = plt.subplots(len(subimgs), 1, figsize=(8, len(subimgs)*2))

if len(subimgs) == 1:
    axes = [axes]  # asegurar que siempre sea iterable

for i, (ax, sub) in enumerate(zip(axes, subimgs)):
    ax.imshow(sub, cmap='gray')
    ax.set_title(f"Renglón {i+1}")
    ax.axis('off')

plt.show()




col_sum = np.sum(subimgs, axis=0)
th_col = 0.8 * np.max(col_sum)
line_mask = col_sum > th_col

cols = []
en_linea = False
for i, val in enumerate(line_mask):
    if val and not en_linea:
        x1 = i
        en_linea = True
    elif not val and en_linea:
        x2 = i
        cols.append((x1, x2))
        en_linea = False

# --- Generar subimágenes entre líneas verticales ---
respuesta = []
for i in range(len(cols) - 1):
    x1 = cols[i][1]
    x2 = cols[i + 1][0]
    respuesta.append(sub[:, x1:x2])


# --- Mostrar todos los renglones en una sola figura ---
fig, axes = plt.subplots(len(respuesta), 1, figsize=(8, len(respuesta)*2))

if len(respuesta) == 1:
    axes = [axes]  # asegurar que siempre sea iterable

for i, (ax, sub) in enumerate(zip(axes, respuesta)):
    ax.imshow(sub, cmap='gray')
    ax.set_title(f"Renglón {i+1}")
    ax.axis('off')

plt.show()











# --- Detectar renglones (líneas horizontales) ---
row_sum = np.sum(img_bool, axis=1)
th_row = 0.5 * np.max(row_sum)
line_mask = row_sum > th_row

renglones = []
en_linea = False
for i, val in enumerate(line_mask):
    if val and not en_linea:
        y1 = i
        en_linea = True
    elif not val and en_linea:
        y2 = i
        renglones.append((y1, y2))
        en_linea = False

# --- Subimágenes entre líneas horizontales ---
subimgs = []
for i in range(len(renglones) - 1):
    y1 = renglones[i][1]
    y2 = renglones[i + 1][0]
    subimgs.append((y1, y2, img[y1:y2, :]))

# --- Procesar cada renglón ---
for idx, (y1, y2, sub) in enumerate(subimgs):
    sub_bool = sub < th  # binarizamos el renglón

    # --- Detectar columnas en este renglón ---
    col_sum = np.sum(sub_bool, axis=0)
    th_col = 0.8 * np.max(col_sum)
    col_mask = col_sum > th_col

    cols = []
    en_linea = False
    for j, val in enumerate(col_mask):
        if val and not en_linea:
            x1 = j
            en_linea = True
        elif not val and en_linea:
            x2 = j
            cols.append((x1, x2))
            en_linea = False

    # --- Subimágenes de cada celda (columnas del renglón) ---
    celdas = []
    for j in range(len(cols) - 1):
        x1 = cols[j][1]
        x2 = cols[j + 1][0]
        celdas.append(sub[:, x1:x2])

    # --- Mostrar todas las columnas del renglón ---
    fig, axes = plt.subplots(1, len(celdas), figsize=(len(celdas) * 3, 3))
    if len(celdas) == 1:
        axes = [axes]
    for j, (ax, celda) in enumerate(zip(axes, celdas)):
        ax.imshow(celda, cmap='gray')
        ax.set_title(f"R{idx+1}C{j+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()












#### TEST

# --- 6. Aplicar connectedComponentsWithStats a cada renglón ---
lista = []
th_area = 10  # umbral mínimo de área para filtrar ruido

for renglon in subimgs:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(renglon, connectivity=8)
    
    # Filtrar por área mínima
    ix_area = stats[:, -1] > th_area
    stats_filtered = stats[ix_area, :]
    centroids_filtered = centroids[ix_area]

imshow(subimgs[2])




renglones_dict = []
for y1, y2 in renglones:
    renglones_dict.append({
        "img": img_th[y1:y2, :],
        "cord": [y1, 0, y2, img_th.shape[1]]
    })

# Detectar letras
letras = []
il = -1
for ir, renglon in enumerate(renglones_dict):
    # Acondicionamiento
    renglon_zeros = renglon["img"] < 50

    # Detectar inicio-fin de letras
    ren_col_zeros = renglon_zeros.any(axis=0)
    x = np.diff(ren_col_zeros.astype(int))
    letras_indxs = np.argwhere(x).flatten()
    letras_indxs[::2] += 1  # ajustar índices pares
    letras_indxs = letras_indxs.reshape(-1, 2)

    for irl, idxs in enumerate(letras_indxs):
        il += 1
        letras.append({
            "ir": ir+1,
            "irl": irl+1,
            "il": il,
            "cord": [renglon["cord"][0], idxs[0], renglon["cord"][2], idxs[1]],
            "img": renglon["img"][:, idxs[0]:idxs[1]]
        })

# Mostrar bounding boxes
plt.figure(), plt.imshow(img, cmap='gray')
for letra in letras:
    yi = letra["cord"][0]
    xi = letra["cord"][1]
    W = letra["cord"][3] - letra["cord"][1]
    H = letra["cord"][2] - letra["cord"][0]
    rect = Rectangle((xi, yi), W, H, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
plt.show()