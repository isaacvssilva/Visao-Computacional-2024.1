import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 2) Utilizando como base as figuras ‘circle.jpg’ e ‘line.jpg’, forme o desenho de um “boneco
# palito” aplicando uma sequência de transformações geométricas e operações lógicas nas
# imagens, seguindo as regras abaixo. 

# – A figura resultante deve ter um tamanho de 300x300.
# – Use cópias da figura ‘line.jpg’ para os braços, pernas e tronco do boneco.
# – Não redimensione as imagens para criar o tronco e a cabeça.
# – Cada braço deve ter 75% do tamanho do tronco.
# – As pernas devem estar em um ângulo de 90º entre si e devem ter o dobro do tamanho dos
# braços.
# – Posicione o boneco no centro da imagem.


# --------------------------------------------------------------------------------------------------------
filename = sys.argv[2]

path_linha = sys.argv[1]
path_circulo = sys.argv[2]

img_linha = cv2.imread(path_linha)
img_circulo = cv2.imread(path_circulo)

# Convertendo para o espaco RGB
img_linha_rgb = cv2.cvtColor(img_linha, cv2.COLOR_BGR2RGB)
img_circulo_rgb = cv2.cvtColor(img_circulo, cv2.COLOR_BGR2RGB)

# Dimensoes para a imagem linha
width_linha = img_linha_rgb.shape[1]
height_linha = img_linha_rgb.shape[0]

# Dimensoes para a imagem circulo
width_circulo = img_circulo_rgb.shape[1]
height_circulo = img_circulo_rgb.shape[0]

# -------------------------- PERNAS --------------------------------------------
# Rotacao 45 graus para a linha (pernas) 
x_center_linha_pernas = width_linha / 2
y_center_linha_pernas = height_linha / 2

matriz_rotacao_linha_pernas = cv2.getRotationMatrix2D((x_center_linha_pernas, y_center_linha_pernas), 45, 1)
img_rotacionada_linha_pernas = cv2.warpAffine(img_linha_rgb, matriz_rotacao_linha_pernas,(width_linha, height_linha))

# Espelhando linha 45 graus para a perna
img_rotacionada_linha_pernas_espelho = cv2.flip(img_rotacionada_linha_pernas, 0)

# Translation
M_translation_perna = np.float32([[1,0,21],[0,1,-21]])

img_rotacionada_linha_pernas = cv2.warpAffine(img_rotacionada_linha_pernas, M_translation_perna, (width_linha, height_linha))
nova_img_rotacionada_linha_pernas_espelho = cv2.flip(img_rotacionada_linha_pernas, 1)

# cv2.imshow('perna esquerda', img_rotacionada_linha_pernas)
# cv2.imshow('perna direita', nova_img_rotacionada_linha_pernas_espelho)
# ------------------------------------------------------------------------------

# -------------------------- BRACOS --------------------------------------------
# Redimensionando imagem da linha em 75%
fator_escala_braco = 0.75
largura_braco = int(width_linha * fator_escala_braco)
altura_braco = int(height_linha * fator_escala_braco)
img_linha_braco = cv2.resize(img_linha_rgb, (largura_braco, altura_braco))

x_center_linha_braco = width_linha / 2
y_center_linha_braco = height_linha / 2

matriz_rotacao_linha_braco = cv2.getRotationMatrix2D((x_center_linha_braco, y_center_linha_braco), 0, 1)
img_rotacionada_linha_braco = cv2.warpAffine(img_linha_braco, matriz_rotacao_linha_braco,(width_linha, height_linha))

# Espelhando linha 0 graus para o braco
img_rotacionada_linha_braco_espelho = cv2.flip(img_rotacionada_linha_braco, 0)

# Translation
temp_M_translation_braco = np.float32([[1,0,10],[0,1,62]])
temp_img_rotacionada_linha_braco = cv2.warpAffine(img_rotacionada_linha_braco, temp_M_translation_braco, (width_linha, height_linha))
temp_nova_img_rotacionada_linha_braco_espelho = cv2.flip(temp_img_rotacionada_linha_braco, 0)


pos_img_rotacionada_linha_braco = cv2.vconcat([temp_img_rotacionada_linha_braco, temp_nova_img_rotacionada_linha_braco_espelho])
# cv2.imshow('braco esquerdo', pre_img_rotacionada_linha_braco)
# cv2.imshow('braco direito', pre_nova_img_rotacionada_linha_braco_espelho)
# cv2.imshow('braco grosso', pos_img_rotacionada_linha_braco)

M_translation_braco = np.float32([[1,0,35],[0,1,-35]])

img_rotacionada_linha_braco = cv2.warpAffine(pos_img_rotacionada_linha_braco, M_translation_braco, (width_linha, height_linha))
nova_img_rotacionada_linha_braco_espelho = cv2.flip(img_rotacionada_linha_braco, 1)
boneco_palito_braco_final = cv2.hconcat([img_rotacionada_linha_braco, nova_img_rotacionada_linha_braco_espelho])

# cv2.imshow('braco esquerdo', img_rotacionada_linha_braco)
# cv2.imshow('braco direito', nova_img_rotacionada_linha_braco_espelho)
# cv2.imshow('bracos retos', boneco_palito_braco_final)
# ------------------------------------------------------------------------------

# -------------------------- CABECA --------------------------------------------

img_circulo_cabeca = img_circulo_rgb.copy()

M_translation_cabeca = np.float32([[1,0,50],[0,1,28]])

img_circulo_cabeca = cv2.warpAffine(img_circulo_cabeca, M_translation_cabeca, (width_circulo, height_circulo))
img_circulo_cabeca_espelho = cv2.flip(img_circulo_cabeca, 1)

# cv2.imshow('boneco cabeca 1', img_circulo_cabeca)
# cv2.imshow('boneco cabeca 2', img_circulo_cabeca_espelho)

boneco_palito_braco_cabeca = cv2.vconcat([img_circulo_cabeca, img_rotacionada_linha_braco])
boneco_palito_braco_cabeca_espelho = cv2.flip(boneco_palito_braco_cabeca, 1)
# cv2.imshow('boneco cabeca 3', boneco_palito_braco_cabeca)
# cv2.imshow('boneco cabeca 4', boneco_palito_braco_cabeca_espelho)
boneco_palito_braco_cabeca_final = cv2.hconcat([boneco_palito_braco_cabeca, boneco_palito_braco_cabeca_espelho])
#cv2.imshow('boneco cabeca final', boneco_palito_braco_cabeca_final)
# ------------------------------------------------------------------------------

# -------------------------- TRONCO ---------------------------------------------
# Realizando a copia da imagem linha original no espaco RGB
img_linha_tronco = img_linha_rgb.copy()

x_center_linha_tronco = width_linha / 2
y_center_linha_tronco = height_linha / 2

matriz_rotacao_tronco = cv2.getRotationMatrix2D((x_center_linha_tronco, y_center_linha_tronco), 90, 1)
img_rotacionada_linha_tronco = cv2.warpAffine(img_linha_tronco, matriz_rotacao_tronco, (width_linha, height_linha))

M_translation_tronco = np.float32([[1,0,50],[0,1,10]])
temp_M_translation_pernas_final = np.float32([[1,0,0],[0,1,-19]])

img_rotacionada_linha_tronco = cv2.warpAffine(img_rotacionada_linha_tronco, M_translation_tronco, (width_linha, height_linha))
img_rotacionada_linha_tronco_espelho = cv2.flip(img_rotacionada_linha_tronco, 1)

boneco_palito_perna_esquerda = cv2.vconcat([img_rotacionada_linha_tronco, img_rotacionada_linha_pernas])
boneco_palito_perna_direita = cv2.vconcat([img_rotacionada_linha_tronco_espelho, nova_img_rotacionada_linha_pernas_espelho])

temp_boneco_palito_pernas_final = cv2.hconcat([boneco_palito_perna_esquerda, boneco_palito_perna_direita])
#cv2.imshow('temp boneco tronco pernas final', temp_boneco_palito_pernas_final)

boneco_palito_pernas_final = cv2.warpAffine(temp_boneco_palito_pernas_final, temp_M_translation_pernas_final, (temp_boneco_palito_pernas_final.shape[1], temp_boneco_palito_pernas_final.shape[0]))
#cv2.imshow('oficial tronco pernas final', boneco_palito_pernas_final)

# Concatenando a parte a cabeca e os bracos, com o tronco
M_translation_cabeca_final = np.float32([[1,0,0],[0,1,96]])

img_circulo_cabeca_final = cv2.warpAffine(boneco_palito_braco_cabeca_final, M_translation_cabeca_final, (boneco_palito_braco_cabeca_final.shape[1], boneco_palito_braco_cabeca_final.shape[0]))

#cv2.imshow('img_circulo_cabeca_final', img_circulo_cabeca_final)

M_translation_pernas_final = np.float32([[1,0,0],[0,1,-10]])
boneco_palito_pernas_final = cv2.warpAffine(boneco_palito_pernas_final, M_translation_pernas_final, (boneco_palito_pernas_final.shape[1], boneco_palito_pernas_final.shape[0]))
boneco_palito_final = cv2.vconcat([img_circulo_cabeca_final, boneco_palito_pernas_final])

cv2.imshow('boneco palito final', boneco_palito_final)
tamanho_final = (300, 300)
boneco_palito_final_redimensionado = cv2.resize(boneco_palito_final, tamanho_final)

cv2.imshow('Boneco Palito Final 300x300', boneco_palito_final_redimensionado)
# ------------------------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
