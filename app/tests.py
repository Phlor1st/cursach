import sys
import os
import numpy as np

# Добавляем пути
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui_ramanspy import GraphenePredictor


def test_predictor():
    """Тестирует загрузку модели и предсказание"""
    try:
        predictor = GraphenePredictor("models/graphene_model.keras", "models/scalers.npz")

        # Тестовые параметры
        test_cases = [
            (100, 30, "C-Al"),
            (100, 30, "C-Co"),
            (100, 30, "C-Ta")
        ]

        for current, pressure, combination in test_cases:
            result = predictor.predict(current, pressure, combination)
            print(f"✅ {combination}: I_D/I_G={result['id_ig']:.3f}, I_2D/I_G={result['i2d_ig']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


if __name__ == "__main__":
    test_predictor()
