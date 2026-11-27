import streamlit as st
import sys
import os
import traceback
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from predict import GraphenePredictor, ALL_ELEMENT_COMBINATIONS

MIN_CURRENT = 80
MAX_CURRENT = 120
MIN_PRESSURE = 3
MAX_PRESSURE = 60  # торр, как при обучении в neyronka/main.py

MODEL_FILE_PATH = "models/graphene_model_compatible.keras"
SCALERS_PATH = "models/scalers.npz"

@st.cache_resource
def load_predictor():
    """Загружает модель и скейлеры"""
    try:
        if not os.path.exists(MODEL_FILE_PATH):
            st.error(f"Файл модели не найден: {MODEL_FILE_PATH}")
            return None
        if not os.path.exists(SCALERS_PATH):
            st.error(f"Файл скейлеров не найден: {SCALERS_PATH}")
            return None

        predictor = GraphenePredictor(MODEL_FILE_PATH, SCALERS_PATH)
        return predictor
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.code(traceback.format_exc())
        return None


def ui(current, pressure, combination, predictor):
    """Основной интерфейс приложения"""
    st.title("Предсказатель параметров графена")

    if predictor:
        try:
            result = predictor.predict(current, pressure, combination)

            # Метрики
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("I_D/I_G", f"{result['id_ig']:.3f}")
            with col2:
                st.metric("I_2D/I_G", f"{result['i2d_ig']:.3f}")
            with col3:
                st.metric("Тип материала", f"{result['material_code']}")

            # Характеристики
            st.info(f"Дефектность: {result['defect']}")
            st.success(f"Слойность: {result['layers']}")

            # Дополнительная информация
            with st.expander("Детали предсказания"):
                st.write("Параметры синтеза:")
                st.write(f"- Ток: {current} А")
                st.write(f"- Давление: {pressure} торр")
                st.write(f"- Материал: {combination} (тип {result['material_code']})")
                st.write(f"- I_D/I_G: {result['id_ig']:.3f}")
                st.write(f"- I_2D/I_G: {result['i2d_ig']:.3f}")

        except Exception as e:
            st.error(f"Ошибка во время предсказания: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("Не удалось загрузить модель. Проверьте наличие файлов модели и скейлеров.")


# Основное приложение Streamlit
def main():
    st.set_page_config(layout="wide", page_title="Предсказатель графена")

    # Сайдбар с параметрами
    st.sidebar.header("Параметры процесса")

    current = st.sidebar.slider(
        'Сила тока (А)',
        min_value=float(MIN_CURRENT),
        max_value=float(MAX_CURRENT),
        value=100.0,
        step=1.0
    )

    pressure = st.sidebar.slider(
        'Давление (торр)',
        min_value=float(MIN_PRESSURE),
        max_value=float(MAX_PRESSURE),
        value=30.0,
        step=0.1
    )

    combination = st.sidebar.selectbox(
        'Материал подложки',
        options=ALL_ELEMENT_COMBINATIONS,
    )

    # Загружаем predictor
    predictor = load_predictor()

    # Кнопка предсказания
    if st.sidebar.button('Предсказать'):
        ui(current, pressure, combination, predictor)
    else:
        st.info("Нажмите кнопку 'Предсказать' на панели слева, чтобы запустить модель.")


if __name__ == "__main__":
    main()
