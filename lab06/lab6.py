import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Modificar o para utilizar os descritores SURF e ORB.

def load_image(image_path, color_mode=cv2.IMREAD_GRAYSCALE):
    """Carregando a imagem do caminho especificado."""
    image = cv2.imread(image_path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem de {image_path}")
    return image

def detect_and_compute_features(image, method='SURF', hessian_threshold=400):
    """Detectando keypoints e calculando descritores usando SURF ou ORB."""
    if method == 'SURF':
        feature_detector = cv2.xfeatures2d_SURF.create(hessian_threshold)
    elif method == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError(f"Método {method} desconhecido. Escolha 'SURF' ou 'ORB'.")
    
    keypoints, descriptors = feature_detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, method='SURF'):
    """Encontrando correspondências entre descritores usando BFMatcher."""
    if method == 'SURF':
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    elif method == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        good_matches = sorted(matches, key=lambda x: x.distance)
    else:
        raise ValueError(f"Método {method} desconhecido. Escolha 'SURF' ou 'ORB'.")
    
    return good_matches

def draw_feature_matches(img1, img2, kp1, kp2, matches, method='SURF', max_matches=15):
    """Desenhando correspondências de características entre duas imagens."""
    if method == 'SURF':
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif method == 'ORB':
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        raise ValueError(f"Método {method} desconhecido. Escolha 'SURF' ou 'ORB'.")
    
    return match_img

def display_side_by_side(images, titles):
    """Exibindo imagens lado a lado com títulos."""
    plt.figure(figsize=(20, 10))
    for idx, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), idx)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.show()

def main(image_path1, image_path2):
    # Carregando as imagens em escala de cinza
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    # Processamento e correspondências usando SURF
    kp1_surf, des1_surf = detect_and_compute_features(image1, method='SURF')
    kp2_surf, des2_surf = detect_and_compute_features(image2, method='SURF')
    surf_matches = match_features(des1_surf, des2_surf, method='SURF')
    surf_match_image = draw_feature_matches(image1, image2, kp1_surf, kp2_surf, surf_matches, method='SURF')

    # Processamento e correspondências usando ORB
    kp1_orb, des1_orb = detect_and_compute_features(image1, method='ORB')
    kp2_orb, des2_orb = detect_and_compute_features(image2, method='ORB')
    orb_matches = match_features(des1_orb, des2_orb, method='ORB')
    orb_match_image = draw_feature_matches(image1, image2, kp1_orb, kp2_orb, orb_matches, method='ORB')

    # Salvando as imagens de correspondência
    cv2.imwrite('output_surf.jpg', surf_match_image)
    cv2.imwrite('output_orb.jpg', orb_match_image)

    # Exibindo as imagens lado a lado
    display_side_by_side([surf_match_image, orb_match_image], ['SURF Matches', 'ORB Matches'])

if __name__ == "__main__":
    # Verificando se os caminhos das imagens foram fornecidos como argumentos
    if len(sys.argv) < 3:
        print("Uso: python lab3.py <caminho_imagem1> <caminho_imagem2>")
        sys.exit(1)

    # Caminhos das imagens fornecidos via linha de comando
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    # Chamando a função principal com os caminhos das imagens
    main(img1_path, img2_path)
