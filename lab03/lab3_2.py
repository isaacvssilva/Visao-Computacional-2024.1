import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_image(path):
    """ Carregando a imagem do caminho especificado em escala de cinza. """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")
    return image

def perform_fft(image):
    """ Realizando a transformada de Fourier e deslocando o quadrante zero para o centro. """
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(dft)

def create_filter_mask(dims, filter_type, cutoff):
    """ Criando uma máscara de filtro no domínio da frequência. """
    rows, cols = dims
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if filter_type == 'lowpass' and distance <= cutoff:
                mask[i, j] = 1
            elif filter_type == 'highpass' and distance > cutoff:
                mask[i, j] = 1
    return mask

def apply_filter(dft_shift, mask):
    """ Aplicando a máscara ao deslocamento DFT e realizando a inversão da FFT. """
    fshift_masked = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

def process_image(path, filter_type, cutoff):
    """ Processo completo desde carregar até filtrar a imagem. """
    img = load_image(path)
    dft_shifted = perform_fft(img)
    mask = create_filter_mask(img.shape, filter_type, cutoff)
    filtered_img = apply_filter(dft_shifted, mask)
    return filtered_img

def display_results(images, titles):
    """ Exibindo imagens com títulos. """
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def main():
    paths = ['images/halftone.png', 'images/pieces.png', 'images/salt_noise.png']
    filters = [('lowpass', 50), ('highpass', 30), ('lowpass', 40)]
    processed_images = []
    titles = []

    for path, (ftype, cutoff) in zip(paths, filters):
        processed_img = process_image(path, ftype, cutoff)
        processed_images.append(processed_img)
        titles.append(f'Filtrado ({ftype})')
        titles.append('Original')
        processed_images.append(load_image(path))

    display_results(processed_images, titles)

if __name__ == "__main__":
    main()
