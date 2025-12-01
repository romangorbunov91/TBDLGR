import numpy as np
import os
import mediapipe as mp
import cv2


def media_pipe(root, pictures, out_dir_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Инициализация модели MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        # Перебираем все изображения
        for idx, file in enumerate(pictures):
            # Читаем изображение
            print(os.path.join(root, file))
            image = cv2.flip(cv2.imread(os.path.join(root, file)), 1)
            
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
            output_filename = os.path.join(root.replace('frames', out_dir_name), f"frame_{idx:03d}.jpg")
            cv2.imwrite(output_filename, cv2.flip(annotated_image, 1))
            print(f"Сохранено размеченное изображение: {output_filename}")


# Пример использования
if __name__ == "__main__":
    # Путь к видео
    
    in_folder = './datasets/Bukva/frames'
    out_folder = 'mediapipe_frames'  # Папка для сохранения размеченных изображений
    '''
    pictures = []
    for root, dirs, files in os.walk(in_folder):
        for file in files:
            print(file)
            
    '''            
    for root, dirs, files in os.walk(in_folder):
        new_root = root.replace('frames', out_folder)
        if not os.path.exists(new_root):
            os.makedirs(new_root)  
        
        #for file in files:
        #full_path = os.path.join(root, file)
        media_pipe(root, files, out_folder)