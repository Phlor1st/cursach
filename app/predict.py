import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- КОМБИНАЦИИ ЭЛЕМЕНТОВ (ВОСПРОИЗВОДИМЫЙ ПОРЯДОК) ---
# ПОРЯДОК ВАЖЕН! Новые комбинации добавлять только в конец.
ALL_ELEMENT_COMBINATIONS = [
    "C-Al",
    "C-Co",
    "C-Ta",
]

# --- АВТОНУМЕРАЦИЯ КОМБИНАЦИЙ (0,1,2 — тот же порядок, что material_type в neyronka/main.py) ---
COMBINATION_TO_INT = {name: i for i, name in enumerate(sorted(list(ALL_ELEMENT_COMBINATIONS)))}
N_COMBINATIONS = len(COMBINATION_TO_INT)


class GraphenePredictor:
    """Класс для загрузки и использования модели графена"""

    def __init__(self, model_path, scalers_path):
        # Используем его маппинг
        self.material_mapping = COMBINATION_TO_INT

        # Загружаем модель
        self.model = keras.models.load_model(model_path)

        # Загружаем скейлеры
        self.scalers = np.load(scalers_path)
        self.scaler_X_mean = self.scalers['scaler_X_mean']
        self.scaler_X_scale = self.scalers['scaler_X_scale']
        self.scaler_y_mean = self.scalers['scaler_y_mean']
        self.scaler_y_scale = self.scalers['scaler_y_scale']

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
        # Используем его маппинг
        material_code = self.material_mapping[combination]

        input_data = np.array([[current, material_code, pressure]], dtype=np.float32)
        input_scaled = (input_data - self.scaler_X_mean) / self.scaler_X_scale

        prediction = self.model.predict(input_scaled, verbose=0)
        prediction_denorm = (prediction * self.scaler_y_scale) + self.scaler_y_mean

        id_ig, i2d_ig = prediction_denorm[0]
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
