import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]

img = cv2.imread(filename)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ------------ TRACKBAR para encontrar os valores de HSV ------------ 
# ------- BEGIN 

# # Função de retorno de chamada que faz nada (necessária para a criação do trackbar)
# def nothing(x):
#     pass

# # Cria janela para a imagem e trackbars
# cv2.namedWindow('image')

# # Cria os trackbars para os valores de Hue, Saturation e Value
# cv2.createTrackbar('H_min', 'image', 0, 179, nothing)
# cv2.createTrackbar('S_min', 'image', 0, 255, nothing)
# cv2.createTrackbar('V_min', 'image', 0, 255, nothing)
# cv2.createTrackbar('H_max', 'image', 179, 179, nothing)
# cv2.createTrackbar('S_max', 'image', 255, 255, nothing)
# cv2.createTrackbar('V_max', 'image', 255, 255, nothing)

# while True:
#     # Lê os valores dos trackbars
#     h_min = cv2.getTrackbarPos('H_min', 'image')
#     s_min = cv2.getTrackbarPos('S_min', 'image')
#     v_min = cv2.getTrackbarPos('V_min', 'image')
#     h_max = cv2.getTrackbarPos('H_max', 'image')
#     s_max = cv2.getTrackbarPos('S_max', 'image')
#     v_max = cv2.getTrackbarPos('V_max', 'image')

#     # Cria a máscara de acordo com os valores dos trackbars
#     lower_hsv = np.array([h_min, s_min, v_min])
#     upper_hsv = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

#     # Aplica a máscara na imagem HSV e converte para o espaço de cores BGR para exibição
#     result_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
#     result_img = cv2.cvtColor(result_img, cv2.COLOR_HSV2BGR)

#     # Exibe a imagem original e a imagem resultante com a máscara aplicada
#     #cv2.imshow('original', img)
#     cv2.imshow('image', result_img)

#     # Aguarda o pressionamento da tecla 'ESC' para sair do loop
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# # Fecha todas as janelas abertas
# cv2.destroyAllWindows()

# ------- END

# Comentar o codigo abaixo caso for usar o trackbar 

# Valores HSV para Gamora (verde)
gamora_lower_hsv = np.array([33, 30, 9])
gamora_upper_hsv = np.array([68, 255, 255])

# Valores HSV para Nebulosa (azul)
nebulosa_lower_hsv = np.array([75, 37, 43])
nebulosa_upper_hsv = np.array([106, 255, 255])

# Criando mascaras para as regioes de cor da Gamora e da Nebulosa
mask_gamora = cv2.inRange(img_hsv, gamora_lower_hsv, gamora_upper_hsv)
mask_nebulosa = cv2.inRange(img_hsv, nebulosa_lower_hsv, nebulosa_upper_hsv)

# Calculando os valores medios de HSV para cada região
gamora_mean_hsv = cv2.mean(img_hsv, mask=mask_gamora)
nebulosa_mean_hsv = cv2.mean(img_hsv, mask=mask_nebulosa)

# Realizando a troca de cores entre as regioes de interesse do matiz e saturacao (HUE e SATURATION)
img_hsv[:, :, 0][mask_gamora > 0] = nebulosa_mean_hsv[0]
img_hsv[:, :, 1][mask_gamora > 0] = nebulosa_mean_hsv[1]

# Realizando a troca de cores entre as regioes de interesse do matiz e saturacao (HUE e SATURATION)
img_hsv[:, :, 0][mask_nebulosa > 0] = gamora_mean_hsv[0]
img_hsv[:, :, 1][mask_nebulosa > 0] = gamora_mean_hsv[1]

# Convertendo para o espaco BGR 
img_bgr_final = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Original', img)
cv2.imshow('Resultado', img_bgr_final)
cv2.waitKey(0)
cv2.destroyAllWindows()