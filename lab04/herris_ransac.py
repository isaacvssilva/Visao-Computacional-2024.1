import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Verificando se o caminho da imagem foi fornecido
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_image>")
    sys.exit(1)

# Lendo o caminho da imagem dos argumentos da linha de comando
image_path = sys.argv[1]

# Carregando a imagem
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found")
    sys.exit(1)

# Fazendo uma cópia da imagem
img_cpy = np.copy(img)

# Convertendo a imagem para escala de cinza
gray_image = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray_image)

# Detectando cantos usando o detector de cantos Harris
herris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.01)
corners = np.argwhere(herris_corners > 0.01 * herris_corners.max())
corners = np.flip(corners, axis=1)  # Convertendo (linha, coluna) para (x, y)

# Função RANSAC para ajuste de linha
def fit_line_ransac(points, threshold=5.0, max_iterations=1000):
    best_line = None
    best_inliers = 0

    for _ in range(max_iterations):
        # Selecionando dois pontos aleatórios
        sample = points[np.random.choice(points.shape[0], 2, replace=False)]
        (x1, y1), (x2, y2) = sample

        # Calculando os parâmetros da linha (ax + by + c = 0)
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # Calculando a distância de todos os pontos para a linha
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)
        inliers = points[distances < threshold]
        num_inliers = len(inliers)

        # Atualizando a melhor linha se esta for melhor
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_line = (a, b, c)

    # Calculando valores de y para os pontos extremos para visualização
    if best_line:
        a, b, c = best_line
        x_vals = np.array([0, img.shape[1]])
        y_vals = -(a * x_vals + c) / b

        # Exibindo a linha ajustada sobre a imagem
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(x_vals, y_vals, 'r')
        plt.title('Best RANSAC Line')
        plt.axis('off')
        plt.show()

    return best_line

# Ajustando a linha usando RANSAC
best_line = fit_line_ransac(corners)
