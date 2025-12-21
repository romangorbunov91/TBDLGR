import cv2
import streamlit as st
import json
import tempfile
import os
from pathlib import Path

from utils.framer import extract_frames
from demo_functions import load_model, predict_gesture, frames_to_video, resize_to_autoplay, autoplay_video

FPS_FOR_PREVIEW = 20
THUMBNAIL_WIDTH = 64
MACRO_BLOCK_SIZE = 16

MODEL_PATH = Path(r".\checkpoints\Bukva\best_train_bukva.pth")
CONFIG_PATH = Path(r".\src\hyperparameters\Bukva\config.json")
LABEL_MAP_PATH = Path(r".\src\datasets\bukva_label_map.csv")

st.title("✌️ Распознаватель РЖЯ (Дактиль) - Режим видеофайла")
st.markdown("""Загрузите видеофайл""")

if 'model' not in st.session_state:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    else:
        with st.spinner("Загружаем модель..."):
            st.session_state.model = load_model(CONFIG_PATH, MODEL_PATH)
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = None

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
else:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    n_frames = config['data']['n_frames']

if not LABEL_MAP_PATH.exists():
    raise FileNotFoundError(f"Label map file not found: {LABEL_MAP_PATH}")

st.sidebar.header("📁 Загрузка видео")

uploaded_file = st.sidebar.file_uploader(
    "Выберите видеофайл (MP4, AVI, MOV)",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv']
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name
    
    st.sidebar.success(f"Файл загружен: {uploaded_file.name}")

    try:
        if st.sidebar.button("🎬 Распознать жест", type="primary"):
            with st.spinner("Извлекаем кадры из видео..."):
                img_set = extract_frames(
                    video_path=temp_video_path,
                    num_frames=n_frames,
                    method='window',
                    resize_flag=True
                )
                
                if img_set:

                    img_set_RGB = []
                    for img in img_set:
                        img_set_RGB.append(resize_to_autoplay(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), MACRO_BLOCK_SIZE))

                    with st.expander("📷 Извлечённые кадры из видео"):
                        cols = st.columns(10)
                        for idx, img in enumerate(img_set_RGB):
                            with cols[idx % 10]:
                                st.image(img, caption=f"Кадр {idx}", width=THUMBNAIL_WIDTH)
                    
                    with st.spinner("🎥 Создаём видео из кадров..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                            tmpfile.write(uploaded_file.read())
                            frame_video_path = tmpfile.name
                        try:
                            frames_to_video(img_set_RGB, frame_video_path, fps=FPS_FOR_PREVIEW)

                            if not os.path.exists(frame_video_path):
                                st.error("Видео не создано")
                                
                            if os.path.getsize(frame_video_path) == 0:
                                st.error("Видео пустое (0 байт)")

                            autoplay_video(frame_video_path)
                        
                        finally:
                            # Удаляем временный файл.
                            if os.path.exists(frame_video_path):
                                os.unlink(frame_video_path)
                
                    if st.session_state.model:
                        with st.spinner("🤖 Распознается жест..."):
                            gesture, confidence, top3_list = predict_gesture(
                                st.session_state.model,
                                img_set,
                                LABEL_MAP_PATH,
                                device_name=config["device"]
                                )
                            st.session_state.recognition_result = (gesture, confidence)
                            
                            st.success(f"""
                            ## 🎯 **Буква: {gesture}**
                            Уверенность: {confidence:.1%}
                            """)
                            
                            if top3_list:
                                st.subheader("🏆 Топ-3 по уверенности")
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
    finally:
        # 🧹 ALWAYS delete the temp file, even if an error occurs
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

if st.session_state.recognition_result:
    gesture, confidence = st.session_state.recognition_result
    st.sidebar.subheader("📊 Результат")
    st.sidebar.metric("Буква", gesture)
    st.sidebar.metric("Уверенность", f"{confidence:.1%}")

st.divider()
st.caption("Система распознавания РЖЯ (Дактиль) | Режим видеофайла")