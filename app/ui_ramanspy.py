import streamlit as st
import sys
import os
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.__params__ import COMBINATION_TO_INT

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MODEL_FILE_PATH = "models/graphene_model.keras"
SCALERS_PATH = "models/scalers.npz"


class GraphenePredictor:
    """Класс для загрузки и использования TensorFlow модели"""

    def __init__(self, model_path, scalers_path):
        # Маппинг материалов в числовые коды
        self.material_mapping = COMBINATION_TO_INT

        # Загружаем скейлеры
        self.scalers = np.load(scalers_path)
        self.scaler_X_mean = self.scalers['scaler_X_mean']
        self.scaler_X_scale = self.scalers['scaler_X_scale']
        self.scaler_y_mean = self.scalers['scaler_y_mean']
        self.scaler_y_scale = self.scalers['scaler_y_scale']

        # Загружаем модель
        self.model = keras.models.load_model(model_path)

        print("Модель успешно загружена!")

    def interpret_graphene_properties(self, id_ig, i2d_ig):
        """Интерпретация свойств графена"""
        # Дефектность
        defect_ranges = [
            (0.0, 0.1, "ОЧЕНЬ НИЗКАЯ ДЕФЕКТНОСТЬ"),
            (0.1, 0.3, "НИЗКАЯ ДЕФЕКТНОСТЬ"),
            (0.3, 0.6, "УМЕРЕННАЯ ДЕФЕКТНОСТЬ"),
            (0.6, 1.0, "ПОВЫШЕННАЯ ДЕФЕКТНОСТЬ"),
            (1.0, 1.5, "ВЫСОКАЯ ДЕФЕКТНОСТЬ"),
            (1.5, float('inf'), "ОЧЕНЬ ВЫСОКАЯ ДЕФЕКТНОСТЬ")
        ]

        # Слойность
        layer_ranges = [
            (1.8, float('inf'), "ОДНОСЛОЙНЫЙ ГРАФЕН"),
            (1.2, 1.8, "ДВУХСЛОЙНЫЙ ГРАФЕН"),
            (0.7, 1.2, "ТРЕХСЛОЙНЫЙ ГРАФЕН"),
            (0.0, 0.7, "ГРАФИТ")
        ]

        defect = next(
            (label for min_val, max_val, label in defect_ranges
             if min_val <= id_ig < max_val),
            "НЕОПРЕДЕЛЕНО"
        )

        layers = next(
            (label for min_val, max_val, label in layer_ranges
             if min_val <= i2d_ig < max_val),
            "НЕОПРЕДЕЛЕНО"
        )

        return defect, layers

    def predict(self, current, pressure, combination):
        """Предсказание параметров графена"""

        # Преобразуем комбинацию материалов в числовой код
        material_code = self.material_mapping[combination]

        # Создаем входной вектор [current, material_type, pressure] - порядок как при обучении!
        input_vector = np.array([[current, material_code, pressure]], dtype=np.float32)

        # Нормализуем входные данные (как при обучении)
        input_normalized = (input_vector - self.scaler_X_mean) / (self.scaler_X_scale + 1e-8)

        # Предсказание
        output_normalized = self.model.predict(input_normalized, verbose=0)

        # Обратное преобразование нормализации
        predicted_params = output_normalized * (self.scaler_y_scale + 1e-8) + self.scaler_y_mean

        # Убедимся, что нет отрицательных значений
        predicted_params = np.maximum(predicted_params, 0)

        id_ig, i2d_ig = predicted_params[0]

        # Интерпретация результатов
        defect, layers = self.interpret_graphene_properties(id_ig, i2d_ig)

        return {
            'id_ig': float(id_ig),
            'i2d_ig': float(i2d_ig),
            'defect': defect,
            'layers': layers,
            'current': current,
            'pressure': pressure,
            'combination': combination,
            'material_code': material_code
        }


@st.cache_resource
def load_predictor():
    """Загружает модель и скейлеры"""
    try:
        # Проверяем существование файлов
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


def ui(current, pressure, combination, predictor=load_predictor()):
    st.title("Raman Spectroscopy")
    if predictor:
        try:
            # Получаем предсказание
            result = predictor.predict(current, pressure, combination)

            # Дополнительная информация
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("I_D/I_G", f"{result['id_ig']:.3f}")
            with col2:
                st.metric("I_2D/I_G", f"{result['i2d_ig']:.3f}")
            with col3:
                st.metric("Код материала", f"{result['material_code']}")

            # Характеристики
            st.info(f"**Дефектность:** {result['defect']}")
            st.success(f"**Слойность:** {result['layers']}")

            # Показываем сырые данные
            with st.expander("Показать численные значения"):
                st.write("Параметры синтеза:")
                st.write(f"- Ток: {current} А")
                st.write(f"- Давление: {pressure} торр")
                st.write(f"- Материал: {combination}")
                st.write(f"- Код материала: {result['material_code']}")

                st.write("Предсказанные параметры графена:")
                st.write(f"- I_D/I_G: {result['id_ig']:.3f}")
                st.write(f"- I_2D/I_G: {result['i2d_ig']:.3f}")
                st.write(f"- Уровень дефектности: {result['defect']}")
                st.write(f"- Слойность: {result['layers']}")

        except Exception as e:
            st.error(f"Ошибка во время предсказания: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("Не удалось загрузить модель. Проверьте наличие файлов модели и скейлеров.")
