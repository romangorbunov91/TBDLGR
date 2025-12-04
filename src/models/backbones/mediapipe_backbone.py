import mediapipe as mp
import torch
import torch.nn as nn
import cv2
import numpy as np
device = torch.device('cuda')

class MediaPipeBackbone(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(MediaPipeBackbone, self).__init__()

        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,   # Для статичных изображений
            max_num_hands=2,         # Максимум 2 руки
            min_detection_confidence=0.5  # Минимум уверенности для детекции
        )
        
        # Полносвязный слой для преобразования 63 признаков в 512
        self.fc = nn.Linear(63, 512)  # Преобразуем 63 признака в 512 для дальнейшего использования в модели
        self.pool = nn.AdaptiveAvgPool2d((1, 512))  # Адаптивный пулинг для 512 признаков
        self.classifier = nn.Linear(512, out_planes)  # Вход 512 признаков

    def forward(self, x):
        """
        Подаем на вход последовательность кадров (n_frames = 40).
        Каждый кадр обрабатывается через MediaPipe для извлечения landmarks.
        """
        batch_landmarks = []

        for i in range(x.shape[0]):  # Проходим по каждому примеру в батче
            video_landmarks = []  # Список для хранения landmarks для каждого кадра
            for j in range(x.shape[1]):  # Проходим по кадрам для одного примера (например, для одного жеста)
                image_rgb = x[i, j].cpu().detach().numpy()  # Извлекаем кадр из последовательности
                
                # Преобразуем изображение в формат RGB для MediaPipe
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

                # Применяем MediaPipe для извлечения landmarks
                results = self.mp_hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        video_landmarks.append(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]))
                else:
                    # Если рук не найдено, добавляем пустой массив
                    video_landmarks.append(np.zeros((21, 3)))  # 21 точка для руки

            # Преобразуем landmarks для одного примера в тензор PyTorch и отправляем на нужное устройство
            video_landmarks = np.array(video_landmarks)
            video_landmarks = video_landmarks.reshape(video_landmarks.shape[0], -1)  # Преобразуем в плоский вектор

            # Преобразуем в тензор PyTorch
            video_landmarks = torch.tensor(video_landmarks).float().to(x.device)
            batch_landmarks.append(video_landmarks)

        # Объединяем все landmarks для батча в один тензор
        batch_landmarks = torch.stack(batch_landmarks)

        # Применяем полносвязный слой для обработки данных
        x = self.fc(batch_landmarks)

        # Применяем пулинг
        x = self.pool(x).squeeze(dim=1)

        # Классификация
        x = self.classifier(x)

        return x


    def _process_mediapipe(self, x):
        """Обрабатывает изображение с помощью MediaPipe Hands"""
        
        # Если x - это тензор PyTorch, и он на GPU, нужно перевести его на CPU для работы с cv2 и MediaPipe
        if isinstance(x, torch.Tensor):
            if x.is_cuda:  # Проверка, если тензор на GPU
                x = x.detach().cpu().numpy()  # Переводим на CPU и затем в NumPy
            else:
                x = x.detach().numpy()  # Если на CPU, просто преобразуем

        # Преобразуем изображение из BGR (OpenCV) в RGB (формат для MediaPipe)
        image_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        # Масштабируем значения пикселей в диапазон [0, 255] и преобразуем в uint8
        image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
        
        # Применяем MediaPipe для извлечения ключевых точек
        results = self.mp_hands.process(image_rgb)

        # Если руки найдены, извлекаем их ключевые точки
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.append(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]))
            
            # Преобразуем в numpy массив и возвращаем
            landmarks = np.array(landmarks)
            landmarks = landmarks.reshape(landmarks.shape[0], -1)  # Преобразуем в плоский вектор (размерность [batch_size, 63])

            # Преобразуем обратно в тензор PyTorch и возвращаем на GPU (если это было на GPU)
            return torch.from_numpy(landmarks).float().to(x.device)  # Используем from_numpy для прямого преобразования
        else:
            # Если рук нет, возвращаем пустой тензор
            return torch.zeros(x.shape[0], 63).to(x.device)  # Заполняем пустыми данными для отсутствующих рук







