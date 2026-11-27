import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import wofz
from numpy.polynomial import Polynomial

# ==================== НАСТРОЙКА ПУТЕЙ ====================
_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path(os.environ.get("SYNTH_INPUT_DIR", str(_ROOT / "data" / "processed")))
OUTPUT_DIR = Path(os.environ.get("SYNTH_OUTPUT_DIR", str(_ROOT / "data" / "synthetic_spectra")))
INPUT_FILE = "synthetic_spectra_interpretation.txt"  # Входной файл
N_SYNTH = 10_000  # Количество генерируемых спектров (10^4)
SEED = 42
np.random.seed(SEED)


# ==========================
# ФУНКЦИИ
# ==========================

def voigt_profile(x, center, amp, sigma, gamma):
    """Возвращает профиль Фойгта"""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def load_reference_spectrum(input_dir):
    """Загружает один спектр для шаблона"""
    input_dir = Path(input_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError("Нет обработанных спектров в INPUT_DIR!")

    ref_path = str(Path(input_dir) / np.random.choice(files))
    try:
        data = np.loadtxt(ref_path, skiprows=1)
    except Exception:
        data = np.loadtxt(ref_path)
    x = data[:, 0]
    y = data[:, 1]
    print(f"Загружен шаблон: {ref_path}")
    return x, y / np.max(y)  # нормализуем интенсивность


def interpret_id_ig_ratio(ratio):
    """Интерпретация отношения I_D/I_G"""
    if ratio < 0.1:
        return "ОЧЕНЬ НИЗКАЯ ДЕФЕКТНОСТЬ (качественный графен)"
    elif ratio < 0.3:
        return "НИЗКАЯ ДЕФЕКТНОСТЬ"
    elif ratio < 0.5:
        return "УМЕРЕННАЯ ДЕФЕКТНОСТЬ"
    elif ratio < 1.0:
        return "ПОВЫШЕННАЯ ДЕФЕКТНОСТЬ"
    elif ratio < 1.5:
        return "ВЫСОКАЯ ДЕФЕКТНОСТЬ"
    else:
        return "ОЧЕНЬ ВЫСОКАЯ ДЕФЕКТНОСТЬ (аморфный углерод)"


def interpret_i2d_ig_ratio(ratio):
    """Интерпретация отношения I_2D/I_G"""
    if ratio >= 2.0:
        return "ОДНОСЛОЙНЫЙ ГРАФЕН"
    elif ratio >= 1.5:
        return "ДВУХСЛОЙНЫЙ ГРАФЕН"
    elif ratio >= 0.8:
        return "ТРЕХСЛОЙНЫЙ ГРАФЕН"
    elif ratio >= 0.1:
        return "ГРАФИТ/МНОГОСЛОЙНЫЙ ГРАФЕН"
    else:
        return "АМОРФНЫЙ УГЛЕРОД (2D-пик отсутствует)"


def calculate_peak_ratios(params):
    """Вычисление отношений интенсивностей пиков"""
    # Интегральная интенсивность ≈ площадь под кривой
    # Для профиля Фойгта можно аппроксимировать как amp * (sigma + gamma)
    I_D = params["D"]["amp"] * (params["D"]["sigma"] + params["D"]["gamma"])
    I_G = params["G"]["amp"] * (params["G"]["sigma"] + params["G"]["gamma"])
    I_2D = params["2D"]["amp"] * (params["2D"]["sigma"] + params["2D"]["gamma"])

    id_ig_ratio = I_D / I_G if I_G > 0 else 0
    i2d_ig_ratio = I_2D / I_G if I_G > 0 else 0

    return id_ig_ratio, i2d_ig_ratio


def generate_synthetic_spectrum(x):
    """Создаёт реалистичный спектр с тремя Voigt-пиками + фоном + шумом"""

    # ====== Реалистичные физические параметры ======
    base_amp = np.random.uniform(800, 1500)

    # Генерация параметров с контролем соотношений
    amp_G = base_amp
    amp_D = amp_G * np.random.uniform(0.1, 1.0)  # D ≤ G
    amp_2D = amp_G * np.random.uniform(0.5, 2.0)  # 2D ~ 0.5–2 × G

    params = {
        "D": {
            "center": np.random.normal(1350, 4),
            "amp": amp_D,
            "sigma": np.random.uniform(10, 25),
            "gamma": np.random.uniform(10, 25)
        },
        "G": {
            "center": np.random.normal(1580, 3),
            "amp": amp_G,
            "sigma": np.random.uniform(10, 25),
            "gamma": np.random.uniform(10, 25)
        },
        "2D": {
            "center": np.random.normal(2700, 5),
            "amp": amp_2D,
            "sigma": np.random.uniform(15, 35),
            "gamma": np.random.uniform(15, 35)
        }
    }

    # ====== Суммируем пики ======
    y = np.zeros_like(x)
    for key in params:
        y += voigt_profile(
            x,
            center=params[key]["center"],
            amp=params[key]["amp"],
            sigma=params[key]["sigma"],
            gamma=params[key]["gamma"]
        )

    # ====== Добавляем слабый флуоресцентный фон ======
    poly_deg = np.random.choice([1, 2])
    coeffs = np.random.uniform(-1e-5, 1e-5, size=poly_deg + 1)
    baseline = Polynomial(coeffs)(x)
    y += baseline

    # ====== Добавляем реалистичный шум ======
    signal_max = np.max(y)
    snr = np.random.uniform(60, 120)
    noise_scale = max(signal_max / snr, 1e-6)
    noise = np.random.normal(0, noise_scale, size=len(y))
    y_noisy = y + noise

    # ====== Нормировка ======
    y_noisy -= np.min(y_noisy)
    y_noisy *= 3000 / np.max(y_noisy)

    # ====== Вычисление отношений интенсивностей ======
    id_ig_ratio, i2d_ig_ratio = calculate_peak_ratios(params)

    return y_noisy, id_ig_ratio, i2d_ig_ratio


def create_interpretation_file(interpretation_data, output_dir):
    """Создает файл с интерпретацией результатов"""
    interpretation_path = os.path.join(output_dir, "synthetic_spectra_interpretation.txt")

    with open(interpretation_path, 'w', encoding='utf-8') as f:
        f.write("ИНТЕРПРЕТАЦИЯ СИНТЕТИЧЕСКИХ RAMAN-СПЕКТРОВ\n")
        f.write("=" * 60 + "\n")
        f.write(f"Всего спектров: {len(interpretation_data)}\n")
        f.write("=" * 60 + "\n\n")

        for data in interpretation_data:
            f.write(f"Образец: {data['filename']}\n")
            f.write(f"I_D/I_G = {data['id_ig']:.3f} - {data['id_ig_interpretation']}\n")
            f.write(f"I_2D/I_G = {data['i2d_ig']:.3f} - {data['i2d_ig_interpretation']}\n")
            f.write("-" * 50 + "\n")

        # Статистика
        f.write("\nСТАТИСТИЧЕСКАЯ СВОДКА:\n")
        f.write("=" * 40 + "\n")

        # Статистика по дефектности
        defect_levels = [d['id_ig_interpretation'] for d in interpretation_data]
        defect_stats = pd.Series(defect_levels).value_counts()
        f.write("\nРаспределение по дефектности:\n")
        for level, count in defect_stats.items():
            f.write(f"  {level}: {count} спектров ({count / len(interpretation_data) * 100:.1f}%)\n")

        # Статистика по типам материалов
        material_types = [d['i2d_ig_interpretation'] for d in interpretation_data]
        material_stats = pd.Series(material_types).value_counts()
        f.write("\nРаспределение по типам материалов:\n")
        for material, count in material_stats.items():
            f.write(f"  {material}: {count} спектров ({count / len(interpretation_data) * 100:.1f}%)\n")

    print(f"Файл интерпретации создан: {interpretation_path}")
    return interpretation_path


# ==========================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================

def main():
    print("=== ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ RAMAN-СПЕКТРОВ ===")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y_ref = load_reference_spectrum(INPUT_DIR)

    interpretation_data = []

    for i in range(N_SYNTH):
        y_syn, id_ig_ratio, i2d_ig_ratio = generate_synthetic_spectrum(x)
        name = f"spectrum_{i + 1:05d}.txt"
        out_path = str(out_dir / name)

        # Сохраняем спектр
        df = pd.DataFrame({"Raman_shift": x, "Intensity": y_syn})
        df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

        # Сохраняем данные для интерпретации
        interpretation_data.append({
            'filename': name,
            'id_ig': id_ig_ratio,
            'i2d_ig': i2d_ig_ratio,
            'id_ig_interpretation': interpret_id_ig_ratio(id_ig_ratio),
            'i2d_ig_interpretation': interpret_i2d_ig_ratio(i2d_ig_ratio)
        })

        if (i + 1) % 500 == 0 or i == 0:
            print(f"Создано: {i + 1}/{N_SYNTH}")

    # Создаем файл интерпретации
    interpretation_path = create_interpretation_file(interpretation_data, str(out_dir))

    # Сохраняем также в CSV для удобства анализа
    interpretation_df = pd.DataFrame(interpretation_data)
    interpretation_csv_path = str(out_dir / "synthetic_spectra_interpretation.csv")
    interpretation_df.to_csv(interpretation_csv_path, index=False, encoding='utf-8')

    print(f"\n✅ Генерация завершена! Всего спектров: {N_SYNTH}")
    print(f"Спектры сохранены в: {out_dir}")
    print(f"Файл интерпретации: {interpretation_path}")
    print(f"CSV с интерпретацией: {interpretation_csv_path}")


# ==========================
# ЗАПУСК
# ==========================
if __name__ == "__main__":
    main()
