import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Deixar a imagem do arquivo 'jato.jpg' mais amarelada usando transformação gamma

# --------------------------------------------------------------------------------------------------------
filename = sys.argv[1]

img = cv2.imread(filename)

# 1) Deixar a imagem do arquivo 'jato.jpg' mais amarelada usando transformação gamma

# Convertendo imagem de entrada para RGB, a fim de manipular os canais
img_jato = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertendo para float e evitando overflow (transbordamento de pixels)
img_jato_float = img_jato.astype(np.float32)

# Aumentando a instensidade do R e G, resultando em tons amarelados e aplicando em todos os pixels
img_jato_amarelada = cv2.normalize(img_jato_float * [2, 2, 1], None, 0, 255, cv2.NORM_MINMAX)

# Convertendo de volta para int8
img_jato_amarelada = img_jato_amarelada.astype(np.uint8)

# Convertendo a imagem resultante para BGR
img_jato_amarelada = cv2.cvtColor(img_jato_amarelada, cv2.COLOR_RGB2BGR)

cv2.imshow('Original', img)
cv2.imshow('Amarelada', img_jato_amarelada)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------------------------------------------------------------