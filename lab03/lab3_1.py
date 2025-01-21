import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    """ Carregando uma imagem do caminho especificado. """
    image = cv2.imread(path, 0)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem de {path}")
    return image

def apply_sobel_filter(image, kernel_size=7):
    """ Aplicando o filtro Sobel nas direções x e y e combinando os resultados. """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

def apply_gaussian_blur(image, kernel_size=(13, 13)):
    """ Aplicando um filtro Gaussiano para suavizar a imagem. """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_median_blur(image, kernel_size=5):
    """ Aplicando um filtro de mediana para reduzir o ruído. """
    return cv2.medianBlur(image, kernel_size)

def enhance_image(original_image, edge_image, alpha=0.15):
    """ Realçando a imagem original com as bordas detectadas. """
    return cv2.addWeighted(original_image, 1, np.uint8(edge_image), alpha, 0)

def save_image(image, filename):
    """ Salvando a imagem no arquivo especificado. """
    cv2.imwrite(filename, image)

def display_images(images, titles):
    """ Exibindo as imagens com títulos correspondentes. """
    plt.figure(figsize=(18, 6))
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def main():
    path1 = 'images/halftone.png'
    path2 = 'images/pieces.png'
    path3 = 'images/salt_noise.png'

    img1 = load_image(path1)
    img2 = load_image(path2)
    img3 = load_image(path3)

    sobel_img2 = apply_sobel_filter(img2)
    img2_enhanced = enhance_image(img2, sobel_img2)

    gaussian_img1 = apply_gaussian_blur(img1)
    median_img3 = apply_median_blur(img3)

    save_image(gaussian_img1, './out/car_filtrado.png')
    save_image(median_img3, './out/board_filtrado.png')
    save_image(img2_enhanced, './out/pieces_filtrado.png')

    display_images([gaussian_img1, median_img3, img2_enhanced], ["Car Gaussian", "Board Median", "Pieces Enhanced"])

if __name__ == "__main__":
    main()
