import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def verificar(img):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) 
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Imagen original")

    th = 87
    img_th = np.uint8(img > th) * 255
    axes[1].imshow(img_th, cmap='gray')
    axes[1].set_title("Umbralado manual")

    plt.tight_layout()
    plt.show()


    img_bool = img_th == 0

    # --- Detectar renglones (líneas horizontales) ---
    row_sum = np.sum(img_bool, axis=1)
    th_row = 0.9 * np.max(row_sum)
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

    # --- Procesar cada renglón y almacenar celdas ---
    celdas = []
    for idx, (y1, y2, sub) in enumerate(subimgs):
        sub_bool = sub < th

        # --- Detectar columnas en este renglón ---
        col_sum = np.sum(sub_bool, axis=0)
        th_col = 0.9 * np.max(col_sum)
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

        # --- Subimágenes de cada celda ---
        celdas_renglon = []
        for j in range(len(cols) - 1):
            x1 = cols[j][1]
            x2 = cols[j + 1][0]
            celdas_renglon.append(sub[:, x1:x2])
        
        celdas.append(celdas_renglon)

    cant_letras = [[0]*len(cr) for cr in celdas]
    cant_palabras = [[0]*len(cr) for cr in celdas]

    n_rows = len(celdas)
    max_cols = max((len(r) for r in celdas), default=0)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max(1, max_cols) * 3, max(1, n_rows) * 3))

    # normalizar shape de axes para indexarlo como axes[row, col]
    if n_rows == 1 and max_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif max_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, celdas_renglon in enumerate(celdas):
        for j, celda in enumerate(celdas_renglon):
            ax = axes[idx, j]
            _, celda_bool = cv2.threshold(celda, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            margen = 3  
            h, w = celda_bool.shape
            celda_bool = celda_bool[margen:h-margen, margen:w-margen]
            
            kernel = np.ones((2, 1), np.uint8)  
            celda_bool = cv2.erode(celda_bool, kernel, iterations=1)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bool, 8, cv2.CV_32S)

            areas = stats[:, cv2.CC_STAT_AREA]
            widths = stats[:, cv2.CC_STAT_WIDTH]
            heights = stats[:, cv2.CC_STAT_HEIGHT]

            ix_valido = np.ones(len(stats), dtype=bool)
            ix_valido[0] = False 

            for i in range(1, len(stats)):
                area = areas[i]
                w = widths[i]
                h = heights[i]
                aspect_ratio = h / w if w > 0 else 0
                
                if area < 3:
                    ix_valido[i] = False
                elif aspect_ratio > 10 and w < 4 and area < 30:
                    ix_valido[i] = False

            stats_filtered = stats[ix_valido, :]
            labels_filtered = np.where(ix_valido)[0]

            # Almacenamos cantidad de letras
            cant_letras[idx][j] = len(labels_filtered)
            
            # Almacenamos cantidad de palabras
            letras_ordenadas = sorted(stats_filtered, key=lambda s: s[cv2.CC_STAT_LEFT])
            palabras = 1
            for i in range(len(letras_ordenadas)):
                distancia = letras_ordenadas[i][cv2.CC_STAT_LEFT] - (letras_ordenadas[i-1][cv2.CC_STAT_LEFT] + letras_ordenadas[i-1][cv2.CC_STAT_WIDTH])
                if distancia > 8:  # umbral de separación entre palabras
                    palabras += 1
            cant_palabras[idx][j] = palabras if len(letras_ordenadas) > 0 else 0

            celda_recortada = celda[margen:celda.shape[0]-margen, margen:celda.shape[1]-margen]
            celda_color = cv2.cvtColor(celda_recortada, cv2.COLOR_GRAY2BGR)

            letras = []
            for idx_comp in range(len(labels_filtered)):
                x, y, w, h, area = stats_filtered[idx_comp]
                cv2.rectangle(celda_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
                letra = celda_bool[y:y+h, x:x+w]
                letras.append((x, letra))

            letras.sort(key=lambda l: l[0])

            ax.imshow(celda_color[..., ::-1])
            ax.set_title(f"R{idx+1}C{j+1}")
            ax.axis('off')
            
    for i in range(n_rows):
        used_cols = len(celdas[i]) if i < len(celdas) else 0
        for j in range(used_cols, max_cols):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


    #####################################################
    #####            VERIFICACIONES
    #####################################################
    checks = {}

    # Nombre y apellido
    na_letras = cant_letras[1][1]
    na_palabras = cant_palabras[1][1]

    checks['Nombre y apellido'] = 'OK' if na_letras <= 25 and na_palabras >= 2 else 'MAL'


    # Edad
    edad_letras = cant_letras[2][1]
    edad_palabras = cant_palabras[2][1]

    checks['Edad'] = 'OK' if edad_letras <= 3 and edad_letras >= 2 and edad_palabras == 1 else 'MAL'


    # Mail
    mail_letras = cant_letras[3][1]
    mail_palabras = cant_palabras[3][1]

    checks['Mail'] = 'OK' if mail_letras <= 25 and mail_palabras == 1 else 'MAL'


    # Legajo
    leg_letras = cant_letras[4][1]
    leg_palabras = cant_palabras[4][1]

    checks['Legajo'] = 'OK' if leg_letras == 8 and leg_palabras == 1 else 'MAL'


    # Preguntas
    for i in range(1, 4): 
        pregunta1_letras = cant_letras[i + 5][1]
        pregunta1_palabras = cant_palabras[i + 5][1]

        pregunta2_letras = cant_letras[i + 5][2]
        pregunta2_palabras = cant_palabras[i + 5][2]

        if ((pregunta1_palabras != 0 and pregunta2_palabras == 0) or (pregunta1_palabras == 0 and pregunta2_palabras != 0)) and (pregunta1_letras <= 1 and pregunta2_letras <= 1):
            checks[f'Pregunta {i}'] = 'OK' 
        else:
            checks[f'Pregunta {i}'] = 'MAL' 


    # Comentarios
    com_letras = cant_letras[9][1]
    com_palabras = cant_palabras[9][1]

    checks['Comentarios'] = 'OK' if com_letras <= 25 and com_palabras >= 1 else 'MAL'


    return checks


imgs = ['formulario_01.png', 'formulario_02.png', 'formulario_03.png', 'formulario_04.png', 'formulario_05.png', 'formulario_vacio.png']

resultados = []
for img in imgs:
    resultados.append(verificar(img))

df = pd.DataFrame(resultados)

df.to_csv("personas.csv", encoding="utf-8")

print(df)