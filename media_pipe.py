import numpy as np
import os
import mediapipe as mp
import cv2


def media_pipe(pictures, out_dir_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Создаём папку для выходных изображений, если её нет
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    # Инициализация модели MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        # Перебираем все изображения
        for idx, file in enumerate(pictures):
            # Читаем изображение
            image = cv2.flip(cv2.imread(file), 1)
            
            # Обрабатываем изображение с помощью MediaPipe
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            
            # Рисуем landmarks на изображении
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Сохраняем размеченное изображение в выходной папке
            output_filename = os.path.join(out_dir_name, f"annotated_{idx+1}.png")
            cv2.imwrite(output_filename, cv2.flip(annotated_image, 1))
            print(f"Сохранено размеченное изображение: {output_filename}")


# Пример использования
if __name__ == "__main__":
    # Путь к видео
    
    output_folder = 'sharp_frames'  # Папка для сохранения самых резких кадров
    annotated_folder = 'annotated_images'  # Папка для сохранения размеченных изображений

    folder_stock_path = "C:\\Users\\ivane\\OneDrive\\Documents\\итмо\\TBDLGR\\trimmed"  # Укажите путь к папке
    pictures = []
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            print(file)
    # media_pipe(file, annotated_folder)
