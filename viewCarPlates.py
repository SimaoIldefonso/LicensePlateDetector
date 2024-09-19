import torch
import cv2
import os

# Carregar o modelo YOLOv5 treinado localmente usando torch.hub.load
model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/plate_detector/weights/best.pt', source='local')

# Função para detectar matrículas em imagem
def detect_license_plates(image):
    results = model(image)
    return results

# Função para desenhar as caixas
def draw_boxes(image, results):
    # Pegando os resultados e desenhando as caixas
    for *box, conf, cls in results.xyxy[0]:
        # Desenhar a caixa
        xmin, ymin, xmax, ymax = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        if model.names[int(cls)] == 'matricula' and conf>= 0.25 : 
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Função para processar as imagens de uma pasta
def process_images_from_folder(folder_path):
    photos = os.listdir(folder_path)

    # Analisar cada foto
    for photo in photos:
        photo_path = os.path.join(folder_path, photo)
        image = cv2.imread(photo_path)

        # Redimensionar para maior eficiência
        image_resized = cv2.resize(image, (640, 640))

        # Detecção de placas
        results = detect_license_plates(image_resized)

        # Desenhar as caixas
        draw_boxes(image_resized, results)

        # Mostrar a imagem com as detecções
        cv2.imshow('License Plate Detection', image_resized)

        # Pressionar 'q' para sair
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Função para processar o feed da webcam
def process_webcam(source=0):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar para maior eficiência
        frame_resized = cv2.resize(frame, (640, 640))

        # Detecção de placas
        results = detect_license_plates(frame_resized)

        # Desenhar as caixas
        draw_boxes(frame_resized, results)

        # Mostrar a imagem com as detecções
        cv2.imshow('License Plate Detection', frame_resized)

        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Modo de execução: Webcam ou Imagens
mode = input("Choose mode: '0' for webcam or '1' to process images: ").strip().lower()

if mode == '0':
    process_webcam(source=0)  # Usa a webcam como entrada
elif mode == '1':
    process_images_from_folder('./pics')  # Processa as imagens na pasta 'pics'
else:
    print("Invalid mode! Choose 'webcam' or 'images'.")
