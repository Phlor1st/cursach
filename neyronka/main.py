import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==================== НАСТРОЙКА ПУТЕЙ ====================
_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = str(_ROOT / "neyronka" / "input")
OUTPUT_DIR = str(_ROOT / "neyronka" / "output")
INPUT_FILE = "synthetic_spectra_interpretation.txt"
MODEL_PATH = os.path.join(OUTPUT_DIR, "graphene_model.keras")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scalers.npz")
PLOT_PATH = os.path.join(OUTPUT_DIR, "training_plot.png")
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "synthetic_training_data.csv")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== СОЗДАНИЕ ОБУЧАЮЩИХ ДАННЫХ ====================
def create_training_data(num_samples=10000):
    """Создание синтетических обучающих данных с тремя типами материалов"""
    print("🎯 СОЗДАНИЕ ОБУЧАЮЩИХ ДАННЫХ...")

    # Коды 0,1,2 — как в app/predict.py COMBINATION_TO_INT (sorted: C-Al, C-Co, C-Ta)
    material_properties = {
        0: {'defect_factor': 0.7, 'layer_factor': 1.3, 'name': 'C-Al'},
        1: {'defect_factor': 1.0, 'layer_factor': 1.0, 'name': 'C-Co'},
        2: {'defect_factor': 1.5, 'layer_factor': 0.7, 'name': 'C-Ta'},
    }

    training_data = []
    np.random.seed(42)

    for i in range(num_samples):
        # Случайные параметры процесса
        current = np.random.uniform(80, 120)  # Ток, А
        pressure = np.random.uniform(3, 60)  # Давление, торр
        material_type = np.random.choice([0, 1, 2])

        material_props = material_properties[material_type]

        # БАЗОВЫЕ ЗАВИСИМОСТИ
        # Зависимость от тока: больше ток → больше дефектов
        base_id_ig = 0.1 + (current - 80) / 40 * 0.8  # 0.1-0.9

        # Зависимость от давления: больше давление → лучше слойность
        base_i2d_ig = 0.5 + (pressure - 3) / 57 * 2.0  # 0.5-2.5

        # ВЛИЯНИЕ ТИПА МАТЕРИАЛА:
        id_ig = base_id_ig * material_props['defect_factor']
        i2d_ig = base_i2d_ig * material_props['layer_factor']

        # Добавляем случайный шум
        id_ig += np.random.normal(0, 0.05)
        i2d_ig += np.random.normal(0, 0.1)

        # Ограничиваем физически возможные значения
        id_ig = max(0.05, min(id_ig, 2.0))
        i2d_ig = max(0.3, min(i2d_ig, 3.0))

        training_data.append({
            'current': round(current, 1),
            'pressure': round(pressure, 1),
            'material_type': material_type,
            'material_name': material_props['name'],
            'id_ig': round(id_ig, 3),
            'i2d_ig': round(i2d_ig, 3)
        })

    df = pd.DataFrame(training_data)
    df.to_csv(TRAINING_DATA_PATH, index=False)

    # Анализ данных
    print("📊 АНАЛИЗ СОЗДАННЫХ ДАННЫХ:")
    print("=" * 50)
    for material_type in [0, 1, 2]:
        material_data = df[df['material_type'] == material_type]
        name = material_data['material_name'].iloc[0]

        avg_id_ig = material_data['id_ig'].mean()
        avg_i2d_ig = material_data['i2d_ig'].mean()

        print(f"{material_type}. {name}:")
        print(f"   Средний I_D/I_G: {avg_id_ig:.3f}")
        print(f"   Средний I_2D/I_G: {avg_i2d_ig:.3f}")
        print(f"   Образцов: {len(material_data)}")
        print()

    print(f"💾 Данные сохранены: {TRAINING_DATA_PATH}")
    return df


# ==================== ОБУЧЕНИЕ МОДЕЛИ ====================
def train_model():
    """Обучение нейронной сети"""
    print("🎯 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ...")

    # Создание или загрузка данных (пересоздание при старом кодировании material_type 1–3)
    if not os.path.exists(TRAINING_DATA_PATH):
        df = create_training_data(10000)
    else:
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"📁 Загружены данные: {TRAINING_DATA_PATH}")
        if (
            "material_type" in df.columns
            and df["material_type"].max() >= 3
        ):
            print("Обнаружено старое кодирование материала (1–3). Пересоздаём выборку под коды 0–2.")
            df = create_training_data(10000)

    # Подготовка данных
    X = df[['current', 'material_type', 'pressure']].values
    y = df[['id_ig', 'i2d_ig']].values

    # Нормализация
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"📊 Размеры данных:")
    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка: {X_test.shape[0]} образцов")

    # Создание модели
    model = create_improved_model(input_size=3)

    print("🔧 Архитектура модели:")
    model.summary()

    # Обучение
    print("🚀 НАЧАЛО ОБУЧЕНИЯ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # Сохранение модели и скейлеров
    model.save(MODEL_PATH)
    np.savez(SCALER_PATH,
             scaler_X_mean=scaler_X.mean_,
             scaler_X_scale=scaler_X.scale_,
             scaler_y_mean=scaler_y.mean_,
             scaler_y_scale=scaler_y.scale_)

    print(f"💾 Модель сохранена: {MODEL_PATH}")
    print(f"💾 Скейлеры сохранены: {SCALER_PATH}")
    print("Для Streamlit скопируйте файлы в app/models/: python run_pipeline.py copy-models")

    # Визуализация обучения
    plot_training_history(history)

    # Оценка модели
    evaluate_model(model, X_test, y_test, scaler_y)

    return model, history


def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"📈 График обучения сохранен: {PLOT_PATH}")


def evaluate_model(model, X_test, y_test, scaler_y):
    """Оценка качества модели"""
    # Предсказания
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    # Метрики
    mae_id_ig = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_i2d_ig = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))

    print("\n📊 ОЦЕНКА КАЧЕСТВА МОДЕЛИ:")
    print("=" * 40)
    print(f"MAE I_D/I_G: {mae_id_ig:.4f}")
    print(f"MAE I_2D/I_G: {mae_i2d_ig:.4f}")

    # Примеры предсказаний
    print("\n🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
    print("Ток | Тип | Давление | I_D/I_G (предск) | I_2D/I_G (предск) | I_D/I_G (реальн) | I_2D/I_G (реальн)")
    print("-" * 90)
    for i in range(min(5, len(X_test))):
        true_vals = y_true[i]
        pred_vals = y_pred[i]
        print(f"{X_test[i][0]:.1f} | {int(X_test[i][1])} | {X_test[i][2]:.1f} | "
              f"{pred_vals[0]:.3f} | {pred_vals[1]:.3f} | {true_vals[0]:.3f} | {true_vals[1]:.3f}")


# ==================== ФУНКЦИИ МОДЕЛИ ====================
def create_improved_model(input_size=3):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_size,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(2, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def interpret_graphene_properties(id_ig, i2d_ig):
    if id_ig < 0.1:
        defect = "ОЧЕНЬ НИЗКАЯ ДЕФЕКТНОСТЬ (качественный графен)"
    elif id_ig < 0.3:
        defect = "НИЗКАЯ ДЕФЕКТНОСТЬ"
    elif id_ig < 0.6:
        defect = "УМЕРЕННАЯ ДЕФЕКТНОСТЬ"
    elif id_ig < 1.0:
        defect = "ПОВЫШЕННАЯ ДЕФЕКТНОСТЬ"
    elif id_ig < 1.5:
        defect = "ВЫСОКАЯ ДЕФЕКТНОСТЬ"
    else:
        defect = "ОЧЕНЬ ВЫСОКАЯ ДЕФЕКТНОСТЬ (аморфный углерод)"

    if i2d_ig > 1.8:
        layers = "ОДНОСЛОЙНЫЙ ГРАФЕН"
    elif i2d_ig > 1.2:
        layers = "ДВУХСЛОЙНЫЙ ГРАФЕН"
    elif i2d_ig > 0.7:
        layers = "ТРЕХСЛОЙНЫЙ ГРАФЕН"
    else:
        layers = "ГРАФИТ/МНОГОСЛОЙНЫЙ ГРАФЕН"

    return defect, layers


# ==================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ====================
def predict_graphene_params(current, material_type, pressure):
    """
    Функция для предсказания параметров графена по введенным значениям
    """
    try:
        print(f"\n🔍 Загрузка модели: {MODEL_PATH}")

        # Проверяем существует ли модель
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print("❌ Модель или скейлеры не найдены! Сначала выполните обучение.")
            return None

        # Загрузка модели
        model = keras.models.load_model(MODEL_PATH)

        # Загрузка скейлеров
        scalers = np.load(SCALER_PATH)
        scaler_X_mean = scalers['scaler_X_mean']
        scaler_X_scale = scalers['scaler_X_scale']
        scaler_y_mean = scalers['scaler_y_mean']
        scaler_y_scale = scalers['scaler_y_scale']

        # Подготовка входных данных
        input_data = np.array([[current, material_type, pressure]], dtype=np.float32)

        # Нормализация входных данных
        input_scaled = (input_data - scaler_X_mean) / scaler_X_scale

        # Предсказание
        prediction = model.predict(input_scaled, verbose=0)

        # Денормализация выходных данных
        prediction_denorm = (prediction * scaler_y_scale) + scaler_y_mean

        id_ig, i2d_ig = prediction_denorm[0]
        defect, layers = interpret_graphene_properties(id_ig, i2d_ig)

        print("\n" + "=" * 60)
        print("🎯 РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
        print("=" * 60)
        print(f"📥 ВХОДНЫЕ ПАРАМЕТРЫ:")
        print(f"   - Ток: {current} A (диапазон: 80-120 A)")
        print(f"   - Тип материала (код): {material_type} ({get_material_name(material_type)})")
        print(f"   - Давление: {pressure} торр (диапазон: 3-60 торр)")
        print(f"\n📤 ПРЕДСКАЗАННЫЕ ПАРАМЕТРЫ ГРАФЕНА:")
        print(f"   - I_D/I_G = {id_ig:.3f}")
        print(f"   - I_2D/I_G = {i2d_ig:.3f}")
        print(f"\n📊 ХАРАКТЕРИСТИКИ ГРАФЕНА:")
        print(f"   - Качество: {defect}")
        print(f"   - Слойность: {layers}")
        print("=" * 60)

        return id_ig, i2d_ig, defect, layers

    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")
        return None


def get_material_name(material_type):
    """Получение названия материала по типу (0, 1, 2)"""
    material_names = {
        0: "C-Al (Алюминий) - лучший катализатор",
        1: "C-Co (Кобальт) - хороший катализатор",
        2: "C-Ta (Тантал) - средний катализатор",
    }
    return material_names.get(material_type, "Неизвестный тип")


# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================
def interactive_prediction_mode():
    """
    Интерактивный режим для ввода параметров и предсказания
    """
    print("\n" + "=" * 70)
    print("🔮 ИНТЕРАКТИВНЫЙ РЕЖИМ ПРЕДСКАЗАНИЯ ПАРАМЕТРОВ ГРАФЕНА")
    print("=" * 70)
    print("📋 ДИАПАЗОНЫ ВХОДНЫХ ПАРАМЕТРОВ:")
    print("   - Ток: 80-120 A")
    print("   - Тип материала: 0-2 (0-C-Al, 1-C-Co, 2-C-Ta)")
    print("   - Давление: 3-60 торр")
    print("=" * 70)

    while True:
        print("\n🎯 Введите параметры для предсказания:")
        print("   (или введите 'q' для выхода)")

        try:
            # Ввод тока
            current_input = input("\n⚡ Ток (80-120 A): ").strip()
            if current_input.lower() == 'q':
                break
            current = float(current_input)

            # Проверка диапазона тока
            if not (80 <= current <= 120):
                print("❌ Ошибка: Ток должен быть в диапазоне 80-120 A")
                print("💡 Пример: 100, 95.5, 115.2")
                continue

            # Ввод типа материала
            type_input = input("🔧 Тип материала (0-2): ").strip()
            if type_input.lower() == 'q':
                break
            material_type = int(type_input)

            # Проверка диапазона типа
            if material_type not in [0, 1, 2]:
                print("❌ Ошибка: Тип материала — целое от 0 до 2")
                print("💡 Пример: 0, 1, 2")
                continue

            # Ввод давления
            pressure_input = input("📊 Давление, торр (3-60): ").strip()
            if pressure_input.lower() == 'q':
                break
            pressure = float(pressure_input)

            # Проверка диапазона давления
            if not (3 <= pressure <= 60):
                print("❌ Ошибка: Давление должно быть в диапазоне 3-60 торр")
                print("💡 Пример: 30, 25.5, 45.2")
                continue

            # Выполнение предсказания
            print("\n🔄 Выполняется предсказание...")
            result = predict_graphene_params(current, material_type, pressure)

            if result is None:
                print("❌ Не удалось выполнить предсказание")

            # Спросить о продолжении
            print("\n" + "-" * 50)
            continue_pred = input("🔄 Выполнить еще одно предсказание? (y/n): ").lower()
            if continue_pred != 'y':
                print("\n👋 Выход из программы...")
                break

        except ValueError:
            print("❌ Ошибка ввода! Убедитесь, что вводите корректные числа.")
            print("💡 Для тока и давления используйте числа (например: 100, 95.5)")
            print("💡 Для типа материала используйте целые числа (1, 2, 3)")
        except KeyboardInterrupt:
            print("\n\n👋 Выход из программы...")
            break


# ==================== РЕЖИМ ПРИМЕРОВ ====================
def example_predictions_mode():
    """
    Режим с готовыми примерами для демонстрации
    """
    print("\n" + "=" * 70)
    print("📊 РЕЖИМ ДЕМОНСТРАЦИОННЫХ ПРИМЕРОВ")
    print("=" * 70)

    # Примеры для демонстрации (только 3 типа)
    examples = [
        {"name": "Высококачественный однослойный (C-Al)", "current": 90, "type": 0, "pressure": 50},
        {"name": "Стандартный двухслойный (C-Co)", "current": 100, "type": 1, "pressure": 30},
        {"name": "Многослойный с дефектами (C-Ta)", "current": 115, "type": 2, "pressure": 10},
        {"name": "Оптимальные параметры (C-Al)", "current": 95, "type": 0, "pressure": 40},
        {"name": "Экстремальные условия (C-Ta)", "current": 120, "type": 2, "pressure": 5},
    ]

    print("Доступные примеры:")
    for i, example in enumerate(examples, 1):
        material_name = get_material_name(example['type'])
        print(f"  {i}. {example['name']}")
        print(
            f"     Ток: {example['current']}A, Тип: {example['type']} ({material_name}), Давление: {example['pressure']}торр")

    while True:
        try:
            choice = input("\nВыберите пример (1-5) или 'q' для выхода: ").strip()
            if choice.lower() == 'q':
                break

            example_num = int(choice)
            if 1 <= example_num <= 5:
                example = examples[example_num - 1]
                print(f"\n🔍 Запуск примера: {example['name']}")
                predict_graphene_params(example['current'], example['type'], example['pressure'])
            else:
                print("❌ Выберите число от 1 до 5")

        except ValueError:
            print("❌ Введите число от 1 до 5 или 'q' для выхода")
        except KeyboardInterrupt:
            break


# ==================== ГЛАВНОЕ МЕНЮ ====================
def main_menu():
    """
    Главное меню программы
    """
    print("🎯 НЕЙРОННАЯ СЕТЬ ДЛЯ ПРЕДСКАЗАНИЯ ПАРАМЕТРОВ ГРАФЕНА")
    print(f"🔧 ФРЕЙМВОРК: TENSORFLOW/KERAS")
    print(f"📁 Модель: {MODEL_PATH}")
    print()

    # Проверяем наличие обученной модели
    if not os.path.exists(MODEL_PATH):
        print("❌ Обученная модель не найдена!")
        print("💡 Сначала выполните обучение модели")
        train_option = input("🔄 Обучить модель сейчас? (y/n): ").lower()
        if train_option == 'y':
            train_model()
        else:
            return

    while True:
        print("\n" + "=" * 60)
        print("🏠 ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("1. 🔮 Интерактивный режим (ввод своих параметров)")
        print("2. 📊 Демонстрационные примеры")
        print("3. 🎯 Обучить модель")
        print("4. 🚪 Выход")
        print("=" * 60)

        choice = input("\nВаш выбор (1-4): ").strip()

        if choice == "1":
            interactive_prediction_mode()
        elif choice == "2":
            example_predictions_mode()
        elif choice == "3":
            train_model()
        elif choice == "4":
            print("\n👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор! Введите 1, 2, 3 или 4")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    print(f"🔄 TensorFlow версия: {tf.__version__}")

    # Запускаем главное меню
    main_menu()