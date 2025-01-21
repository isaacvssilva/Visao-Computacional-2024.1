import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Verificando se o caminho da imagem foi fornecido
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_image>")
    sys.exit(1)

# Lendo o caminho da imagem dos argumentos da linha de comando
filename = sys.argv[1]

# Carregando a imagem
img = cv2.imread(filename)
if img is None:
    print("Error: Image not found")
    sys.exit(1)

# Fazendo uma cópia da imagem original para marcar os cantos
img_cpy = np.copy(img)

# Convertendo a imagem para escala de cinza
gray_image = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)

# Convertendo a imagem para float32
gray = np.float32(gray_image)

# Detectando cantos usando o detector de cantos Harris
harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.01)

# Marcando os cantos detectados na imagem
img_cpy[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

# Configurando a visualização das imagens
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB))
plt.title('Cantos Detectados')
plt.axis('off')

# Exibindo as imagens
plt.show()
