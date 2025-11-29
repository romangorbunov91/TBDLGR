import mediapipe as mp
import torch
import torch.nn as nn
import cv2

class MediaPipeBackbone(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(MediaPipeBackbone, self).__init__()

        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,   # Для статичных изображений
            max_num_hands=2,         # Максимум 2 руки
            min_detection_confidence=0.5  # Минимум уверенности для детекции
        )
        
        # Полносвязный слой для классификации на основе извлечённых признаков ПРИ СЧИТЫВАНИИ 1 РУКИ (21 точка * 3 координаты)
        self.fc = nn.Linear(63, 512)  # Преобразуем 63 признака в 512 для дальнейшего использования в модели
        self.pool = nn.AdaptiveAvgPool2d((1, 512))  # Адаптивный пулинг для 512 признаков
        self.classifier = nn.Linear(512, out_planes)  # Вход 63 признака

    def forward(self, x):
        landmarks = self._process_mediapipe(x)  # Извлекаем landmarks с помощью MediaPipe
        landmarks = landmarks.view(landmarks.size(0), -1)  # Разворачиваем в вектор
        x = self.pool(landmarks).squeeze(dim=1)
        x = self.classifier(x)
        return x

    def _process_mediapipe(self, x):
        """Обрабатывает изображение с помощью MediaPipe Hands"""
        # Преобразуем изображение из BGR (OpenCV) в RGB (формат для MediaPipe)
        image_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        # Применяем MediaPipe для извлечения ключевых точек
        results = self.mp_hands.process(image_rgb)
        
        # Если руки найдены, извлекаем их ключевые точки
        if results.multi_hand_landmarks:
            # Для каждой руки извлекаем 21 точку (landmarks)
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.append(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]))
            # Преобразуем в numpy массив и возвращаем
            return torch.tensor(landmarks).float()
        else:
            # Если рук нет, возвращаем пустой тензор
            return torch.zeros(x.size(0), 63)  # Заполняем пустыми данными для отсутствующих рук

