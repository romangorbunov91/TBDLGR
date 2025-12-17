import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import pandas as pd
from utils.framer_functions import extract_frames
from datasets.utils.normalize import normalize
import imgaug.augmenters as iaa

sys.path.insert(0, str(Path(__file__).parent / "src"))
MODEL_PATH = r".\checkpoints\Bukva\best_train_bukva.pth"

st.title("✌️ Распознаватель РЖЯ (Дактиль) - Режим видеофайла")
st.markdown("""
Загрузите видеофайл, и система извлечёт из него 40 последовательных кадров для анализа.
""")

if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'extracted_frames' not in st.session_state:
    st.session_state.extracted_frames = []
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = None
if 'top3_result' not in st.session_state:
    st.session_state.top3_result = None

@st.cache_resource
def load_model():
    try:
        from models.temporal import GestureTransformer
        import json
        with open('src/hyperparameters/Bukva/config.json', 'r') as f:
            config = json.load(f)
        
        backbone = config['network']['backbone']
        n_classes = config['data']['n_classes']
        n_head = config['network']['n_head']
        dropout2d = config['network']['dropout2d']
        dropout1d = config['network']['dropout1d']
        ff_size = config['network']['ff_size']
        n_module = config['network']['n_module']
        pretrained = config['network']['pretrained']

        in_planes = 3
        
        model = GestureTransformer(
            backbone=backbone,
            in_planes=in_planes,
            n_classes=n_classes,
            pretrained=pretrained,
            n_head=n_head,
            dropout_backbone=dropout2d,
            dropout_transformer=dropout1d,
            dff=ff_size,
            n_module=n_module
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict']
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        st.success(f"✅ Модель загружена успешно!")
        return model
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def prepare_frames_for_model(frames):
    clip = list()
    for frame in frames:
        resized = cv2.resize(frame, (224, 224))
        clip.append(resized)
    clip = np.array(clip).transpose(1, 2, 3, 0)
    clip = normalize(clip)
    transforms = iaa.Noop()
    aug_det = transforms.to_deterministic()
    clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)
    clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
    clip = clip.float()
    clip = clip.unsqueeze(0)
    return clip

def predict_gesture(model, frames):
    if model is None or len(frames) == 0:
        st.error("Модель не загружена или нет кадров")
        return "Модель не загружена", 0.0, []
    

    # --- Загрузка текстовых меток ---
    label_mapping_path = "./src/datasets/bukva_label_mapping.csv"  # или укажите полный путь, если нужно
    label_df = pd.read_csv(label_mapping_path)
    # Убедимся, что метки отсортированы по label_encoded
    label_df = label_df.sort_values('label_encoded')
    class_names = label_df['text'].tolist()
    
    try:
        input_tensor = prepare_frames_for_model(frames)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_conf, top3_idx = torch.topk(probabilities, 3)
            top3_predictions = []
            for i in range(3):
                idx_val = top3_idx[0, i].item()
                conf_val = top3_conf[0, i].item()
                gesture_name = class_names[idx_val] if idx_val < len(class_names) else f"Класс {idx_val}"
                top3_predictions.append((gesture_name, conf_val))
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence_value = confidence.item()
            predicted_idx_value = predicted_idx.item()
        
        if predicted_idx_value < len(class_names):
            predicted_gesture = class_names[predicted_idx_value]
        else:
            predicted_gesture = f"Класс {predicted_idx_value}"
        
        return predicted_gesture, confidence_value, top3_predictions
        
    except Exception as e:
        st.error(f"Ошибка при распознавании: {e}")
        return "Ошибка", 0.0, []

st.sidebar.header("📁 Загрузка видео")

uploaded_file = st.sidebar.file_uploader(
    "Выберите видеофайл (MP4, AVI, MOV)",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv']
)

if uploaded_file is not None:
    temp_video_path = f"temp_{uploaded_file.name}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.uploaded_video = temp_video_path
    
    st.sidebar.success(f"Файл загружен: {uploaded_file.name}")
    
    if st.sidebar.button("🎬 Извлечь кадры и распознать жест", type="primary"):
        with st.spinner("Извлекаем кадры из видео..."):
            
            frames_RGB = extract_frames(
                video_path=temp_video_path,
                num_frames=40,
                method='window',
                resize_flag=True
            )
            frames = []
            for frame in frames_RGB:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frames:
                st.session_state.extracted_frames = frames
                
                st.subheader("📷 Извлечённые кадры из видео")
                cols = st.columns(4)
                for idx, frame in enumerate(frames):
                    with cols[idx % 4]:
                        st.image(frame, caption=f"Кадр {idx+1}", width=150)
                
                if 'model' not in st.session_state:
                    with st.spinner("Загружаем модель распознавания..."):
                        st.session_state.model = load_model()
                
                if st.session_state.model:
                    with st.spinner("🤖 Распознаем жест..."):
                        gesture, confidence, top3_list = predict_gesture(st.session_state.model, frames)
                        st.session_state.recognition_result = (gesture, confidence)
                        st.session_state.top3_result = top3_list
                        
                        st.success(f"""
                        ## 🎯 Результат распознавания
                        ### **Жест: {gesture}**
                        Уверенность: {confidence:.1%}
                        """)
                        
                        if top3_list:
                            st.subheader("🏆 Топ-3 альтернативных варианта")
                            cols = st.columns(3)
                            for i, (name, prob) in enumerate(top3_list):
                                with cols[i]:
                                    medal_color = ["#FFD700", "#C0C0C0", "#CD7F32"][i]
                                    st.markdown(f"<h4 style='text-align: center; color: {medal_color};'>{i+1} место</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<h3 style='text-align: center;'>{name}</h3>", unsafe_allow_html=True)
                                    st.progress(float(prob))
                                    st.markdown(f"<p style='text-align: center;'>{prob:.1%}</p>", unsafe_allow_html=True)
                        else:
                            st.info("Не удалось получить альтернативные варианты.")
                else:
                    st.error("Модель не загружена!")
            else:
                st.error("Не удалось извлечь кадры из видео")

if st.session_state.recognition_result:
    gesture, confidence = st.session_state.recognition_result
    st.sidebar.subheader("📊 Результат")
    st.sidebar.metric("Распознанный жест", gesture)
    st.sidebar.metric("Уверенность", f"{confidence:.1%}")

with st.expander("ℹ️ Как это работает"):
    st.markdown("""
    1. **Загрузите видеофайл** с жестом (MP4, AVI, MOV и другие форматы)
    2. **Система извлечёт 40 кадров** из видео
    3. **Кадры обрабатываются** (изменение размера, нормализация)
    4. **Модель анализирует последовательность кадров** и распознаёт жест
    5. **Отображается результат** с указанием уровня уверенности
    """)

if 'model' not in st.session_state:
    with st.spinner("Загружаем модель распознавания..."):
        st.session_state.model = load_model()

st.divider()
st.caption("Система распознавания РЖЯ (Дактиль) | Режим видеофайла")