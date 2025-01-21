import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Corrigir o código para permitir a visualização completa da imagem após a transformação.

def load_and_resize_image(path, scale_factor=2):
    """Carrega e redimensiona a imagem pelo fator especificado."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem de {path}")
    new_dimensions = (int(image.shape[1] // scale_factor), int(image.shape[0] // scale_factor))
    return cv2.resize(image, new_dimensions)

def convert_to_grayscale(image):
    """Converte a imagem para escala de cinza."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_features(image):
    """Detecta keypoints e descritores na imagem usando SIFT."""
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

def match_features(desc1, desc2):
    """Encontra e filtra os melhores matches entre dois conjuntos de descritores."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]

def find_and_transform_image(img1, img2, kp1, kp2, matches):
    """Calcula a homografia e transforma a primeira imagem para se alinhar com a segunda."""
    if len(matches) < 4:
        raise ValueError("Não há keypoints suficientes.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    
    dimensions = (img2.shape[1], img2.shape[0])
    transformed_img = cv2.warpPerspective(img1, H, dimensions)
    return transformed_img

def display_images(image_list, titles):
    """Exibe uma lista de imagens com títulos correspondentes."""
    plt.figure(figsize=(18, 6))
    for i, (image, title) in enumerate(zip(image_list, titles), 1):
        plt.subplot(1, len(image_list), i)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# Main execution logic
path1 = "images/campus_quixada1.png"
path2 = "images/campus_quixada2.png"

img1 = load_and_resize_image(path1)
img2 = load_and_resize_image(path2)

gray_img1 = convert_to_grayscale(img1)
gray_img2 = convert_to_grayscale(img2)

kp1, desc1 = detect_features(gray_img1)
kp2, desc2 = detect_features(gray_img2)

matches = match_features(desc1, desc2)
transformed_img1 = find_and_transform_image(img1, img2, kp1, kp2, matches)

display_images([img1, img2, transformed_img1], ["Imagem 1", "Imagem 2", "Imagem Transformada"])
