import sys
import cv2
import numpy as np
from time import *


min_width = 80  # Largura minima do retangulo
min_height = 80  # Altura minima do retangulo
offset = 6  # Erro permitido entre pixel
line_pos = 550  # Posição da linha de contagem
delay = 60  # FPS do vídeo
detec = []
cars = 0

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)

# Pega o fundo e subtrai do que está se movendo
subtraction = cv2.createBackgroundSubtractorMOG2()


def get_center(x, y, width, height):
    x1 = width // 2
    y1 = height // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


def set_info(detec):
    global cars

    for (x, y) in detec:
        if (line_pos + offset) > y > (line_pos - offset):
            cars += 1
            cv2.line(frame, (25, line_pos), (1200, line_pos), (0, 127, 255), 3)
            detec.remove((x, y))
            print(f"Carros detectados até o momento: {cars}")


def show_info(frame, dilated):
    text = f"Carros: {cars}"
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", dilated)


while True:
    # Pega cada frame do vídeo
    ret, frame = cap.read()

    # Dá um delay entre cada processamento
    secs = float(1 / delay)
    sleep(secs)

    # Pega o frame e transforma para preto e branco
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Faz um blur para tentar remover as imperfeições da imagem
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Faz a subtração da imagem aplicada no blur
    img_sub = subtraction.apply(blur)

    # "Engrossa" o que sobrou da subtração
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    # Cria uma matriz 5x5, em que o formato da matriz entre 0 e 1 forma uma elipse dentro
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Tenta preencher todos os "buracos" da imagem
    dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    outline, img = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, line_pos), (1200, line_pos), (255, 127, 0), 3)

    for (i, c) in enumerate(outline):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_outline = (w >= min_width) and (h >= min_height)

        if not validate_outline:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame, dilated)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap.release()
