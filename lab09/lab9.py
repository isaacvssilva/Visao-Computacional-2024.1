import sys
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from retinaface import RetinaFace  # Importando RetinaFace

def calculate_iou(boxA, boxB):
    """Calcula o Intersection over Union (IoU) entre dois bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxA[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_image(image_path, color_mode=cv2.IMREAD_GRAYSCALE):
    """Carregando a imagem do caminho especificado."""
    image = cv2.imread(image_path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem de {image_path}. Verifique o caminho e a integridade do arquivo.")
    return image

def detect_faces_haar(image, haar_cascade_path):
    """Detecta rostos usando Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(haar_cascade_path)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

def detect_faces_yolo(image, yolo_model):
    """Detecta rostos usando YOLOv8."""
    results = yolo_model(image)
    faces = [box.xyxy for box in results[0].boxes if box.cls == 0]  # cls=0 refere-se a pessoas
    return [tuple(map(int, face[0].tolist())) for face in faces]

def detect_faces_retina(image):
    """Detecta rostos usando RetinaFace."""
    results = RetinaFace.detect_faces(image)
    faces = []
    for key, result in results.items():
        facial_area = result["facial_area"]
        faces.append((facial_area[0], facial_area[1], facial_area[2], facial_area[3]))
    return faces

def draw_boxes(image, boxes, color):
    """Desenha bounding boxes na imagem."""
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

def main(image_path):
    # Carrega a imagem
    image = load_image(image_path, cv2.IMREAD_COLOR)  # Certificando-se de que está carregando em BGR
    
    # Caminho para o classificador Haar Cascade
    haar_cascade_path = './models/haarcascade_frontalface_default.xml'
    
    # Carregando o modelo YOLO
    yolo_model = YOLO("yolov8n.pt")

    # Detecta rostos com Haar Cascade
    haar_faces = detect_faces_haar(image, haar_cascade_path)
    image_haar = image.copy()
    draw_boxes(image_haar, haar_faces, (255, 0, 0))  # Azul para Haar

    # Detecta rostos com YOLO
    yolo_faces = detect_faces_yolo(image, yolo_model)
    image_yolo = image.copy()
    draw_boxes(image_yolo, yolo_faces, (0, 255, 0))  # Verde para YOLO

    # Detecta rostos com RetinaFace
    retina_faces = detect_faces_retina(image)
    image_retina = image.copy()
    draw_boxes(image_retina, retina_faces, (0, 0, 255))  # Vermelho para RetinaFace
    
    # Calcula IoU médio para Haar x YOLO
    iou_values_haar_yolo = []
    for boxA in haar_faces:
        for boxB in yolo_faces:
            iou = calculate_iou(boxA, boxB)
            iou_values_haar_yolo.append(iou)
    avg_iou_haar_yolo = np.mean(iou_values_haar_yolo) if iou_values_haar_yolo else 0

    # Calcula IoU médio para Haar x Retina
    iou_values_haar_retina = []
    for boxA in haar_faces:
        for boxB in retina_faces:
            iou = calculate_iou(boxA, boxB)
            iou_values_haar_retina.append(iou)
    avg_iou_haar_retina = np.mean(iou_values_haar_retina) if iou_values_haar_retina else 0

    # Exibe as imagens com os bounding boxes
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image_haar, cv2.COLOR_BGR2RGB))
    plt.title(f'Haar Cascade\nTotal de Faces: {len(haar_faces)}')
    plt.subplot(132), plt.imshow(cv2.cvtColor(image_yolo, cv2.COLOR_BGR2RGB))
    plt.title(f'YOLOv8\nTotal de Faces: {len(yolo_faces)}')
    plt.subplot(133), plt.imshow(cv2.cvtColor(image_retina, cv2.COLOR_BGR2RGB))
    plt.title(f'RetinaFace\nTotal de Faces: {len(retina_faces)}')
    plt.suptitle(f'IoU Médio (Haar x YOLO): {avg_iou_haar_yolo:.2f} | IoU Médio (Haar x Retina): {avg_iou_haar_retina:.2f}')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python lab9.py <caminho_imagem>")
        sys.exit(1)

    # Caminho da imagem fornecida via linha de comando
    img_path = sys.argv[1]
    main(img_path)
