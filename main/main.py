import os
import glob
import numpy as np
import ramanspy as rp
import pandas as pd
from lmfit.models import VoigtModel, LinearModel
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from config import INPUT_DIR, OUTPUT_DIR, USE_ORIGIN, LASER_WAVELENGTH_NM, ensure_dirs
from physics_metrics import format_physics_summary

if USE_ORIGIN:
    import originpro as op


def load_spectrum_two_column(path: str):
    """Загрузка двух столбцов: с заголовком (первая строка) или без."""
    try:
        data = np.loadtxt(path, skiprows=1)
    except Exception:
        data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Ожидаются ≥2 столбца: {path}")
    return data[:, 0], data[:, 1]


def save_processed_spectra_files(processed_data, output_dir):
    """Сохранение обработанных спектров (без Origin)."""
    for name, _, processed_spec, _, _, _ in processed_data:
        processed_path_txt = os.path.join(output_dir, f"{name}_processed.txt")
        processed_path_csv = os.path.join(output_dir, f"{name}_processed.csv")
        processed_x = processed_spec.spectral_axis
        processed_y = processed_spec.spectral_data[0]

        np.savetxt(
            processed_path_txt,
            np.column_stack([processed_x, processed_y]),
            delimiter="\t",
            header="Raman_shift\tIntensity",
            comments="",
            fmt="%.2f\t%.6f",
        )
        df_processed = pd.DataFrame({"Raman_shift": processed_x, "Intensity": processed_y})
        df_processed.to_csv(processed_path_csv, index=False)
        print(f"Сохранен: {name}_processed.txt и {name}_processed.csv")


def process_with_ramanspy_controlled(raman_shift, intensity):
    """Предобработка спектров с использованием RamanSPy"""
    mask = raman_shift > 100
    raman_shift_cropped = raman_shift[mask]
    intensity_cropped = intensity[mask]

    spectrum = rp.Spectrum(intensity_cropped.reshape(1, -1), raman_shift_cropped)

    # Используем RamanSPy для предобработки
    denoiser = rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)
    baseline_corrector = rp.preprocessing.baseline.IModPoly()

    denoised = denoiser.apply(spectrum)
    baseline_corrected = baseline_corrector.apply(denoised)

    processed_y = baseline_corrected.spectral_data[0]
    final_spectrum = rp.Spectrum(processed_y.reshape(1, -1), baseline_corrected.spectral_axis)

    return final_spectrum


def voigt_peak_fitting(spectrum):
    """Аппроксимация пиков функциями Voigt"""
    x = spectrum.spectral_axis
    y = spectrum.spectral_data[0]

    print(f"\n--- VOIGT АППРОКСИМАЦИЯ ---")
    print(f"Диапазон: {x[0]:.0f}-{x[-1]:.0f} см⁻¹")

    peak_regions = [
        {'name': 'D', 'prefix': 'd_', 'center': 1350, 'range': (1300, 1400), 'sigma_guess': 20},
        {'name': 'G', 'prefix': 'g_', 'center': 1580, 'range': (1500, 1650), 'sigma_guess': 15},
        {'name': 'TwoD', 'prefix': 'td_', 'center': 2700, 'range': (2600, 2750), 'sigma_guess': 25}
    ]

    fitted_peaks = []

    for region in peak_regions:
        mask = (x >= region['range'][0]) & (x <= region['range'][1])
        x_region = x[mask]
        y_region = y[mask]

        if len(x_region) < 10:
            print(f"{region['name']}-пик: недостаточно точек")
            continue

        try:
            peak_model = VoigtModel(prefix=region['prefix'])
            background = LinearModel(prefix='bkg_')
            model = peak_model + background

            params = model.make_params()
            params[f"{region['prefix']}center"].set(
                value=region['center'],
                min=region['range'][0],
                max=region['range'][1]
            )
            params[f"{region['prefix']}amplitude"].set(value=np.max(y_region) * 2, min=0)
            params[f"{region['prefix']}sigma"].set(value=region['sigma_guess'], min=5, max=50)
            params['bkg_slope'].set(value=0, vary=True)
            params['bkg_intercept'].set(value=np.min(y_region), min=0)

            result = model.fit(y_region, params, x=x_region)

            if result.success:
                # Расчет интегральной интенсивности
                x_dense = np.linspace(region['range'][0], region['range'][1], 1000)
                comps = result.eval_components(x=x_dense)
                voigt_component = comps[region['prefix']]
                integral_intensity = trapezoid(voigt_component, x_dense)

                voigt_params = result.params
                center = voigt_params[f"{region['prefix']}center"].value
                sigma = voigt_params[f"{region['prefix']}sigma"].value

                y_fit = result.best_fit
                ss_res = np.sum((y_region - y_fit) ** 2)
                ss_tot = np.sum((y_region - np.mean(y_region)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

                peak_info = {
                    'name': region['name'],
                    'position': center,
                    'integral_intensity': integral_intensity,
                    'sigma': sigma,
                    'r_squared': r_squared,
                    'fwhm': 3.601 * sigma,
                }

                fitted_peaks.append(peak_info)

                print(f" {region['name']}-пик: {center:.1f} см⁻¹")
                print(f" Интегральная интенсивность: {integral_intensity:.4f}")
                print(f" FWHM: {peak_info['fwhm']:.2f} см⁻¹")
                print(f" R² = {r_squared:.4f}")

            else:
                print(f" {region['name']}-пик: ошибка аппроксимации")

        except Exception as e:
            print(f" {region['name']}-пик: {str(e)}")
            continue

    print(f"Успешно аппроксимировано пиков: {len(fitted_peaks)}")
    return fitted_peaks


def calculate_ratios_from_voigt(peaks):
    """Расчет отношений"""
    ratios = {}
    interpretations = {}

    d_peak = next((p for p in peaks if p['name'] == 'D'), None)
    g_peak = next((p for p in peaks if p['name'] == 'G'), None)
    d2_peak = next((p for p in peaks if p['name'] == 'TwoD'), None)

    print(f"Найдены пики: D={d_peak is not None}, G={g_peak is not None}, 2D={d2_peak is not None}")

    # I_D/I_G
    if d_peak and g_peak:
        id_ig = d_peak['integral_intensity'] / g_peak['integral_intensity']
        ratios['I_D/I_G'] = round(id_ig, 3)

        if id_ig < 0.1:
            interpretation = "ОЧЕНЬ НИЗКАЯ ДЕФЕКТНОСТЬ"
        elif id_ig < 0.5:
            interpretation = "НИЗКАЯ ДЕФЕКТНОСТЬ"
        elif id_ig < 1.0:
            interpretation = "УМЕРЕННАЯ ДЕФЕКТНОСТЬ"
        elif id_ig < 1.5:
            interpretation = "ПОВЫШЕННАЯ ДЕФЕКТНОСТЬ"
        else:
            interpretation = "ВЫСОКАЯ ДЕФЕКТНОСТЬ"

        interpretations['I_D/I_G'] = interpretation
        print(f" I_D/I_G = {id_ig:.3f} - {interpretation}")

        phys = format_physics_summary(id_ig, LASER_WAVELENGTH_NM)
        ratios['La_nm'] = phys.get('La_nm')
        ratios['La_note'] = phys.get('La_note', '')
        if phys.get('La_nm') is not None:
            print(f" La (оценка): {phys['La_nm']} нм — {phys['La_note']}")

    # I_2D/I_G
    if d2_peak and g_peak:
        i2d_ig = d2_peak['integral_intensity'] / g_peak['integral_intensity']
        ratios['I_2D/I_G'] = round(i2d_ig, 3)

        if i2d_ig > 2.0:
            interpretation = "ОДНОСЛОЙНЫЙ ГРАФЕН"
        elif i2d_ig > 1.5:
            interpretation = "ДВУХСЛОЙНЫЙ ГРАФЕН"
        elif i2d_ig > 0.8:
            interpretation = "ТРЕХСЛОЙНЫЙ ГРАФЕН"
        else:
            interpretation = "ГРАФИТ"

        interpretations['I_2D/I_G'] = interpretation
        print(f"I_2D/I_G = {i2d_ig:.3f} - {interpretation}")
    else:
        print("I_2D/I_G не рассчитано: отсутствует 2D-пик или G-пик")

    return ratios, interpretations


def create_origin_plots(processed_data, output_dir, origin_enabled: bool = True):
    """Создание графиков в Origin (спектры уже сохранены в файлы)."""
    if not origin_enabled:
        return
    try:
        op.set_show(True)

        # Книга для спектров
        wb_combined = op.new_book()
        wb_combined.lname = "Все обработанные спектры"

        # Книга для отношений
        wb_ratios = op.new_book()
        wb_ratios.lname = "Отношения интенсивностей"

        # Общая ось X для корректного отображения
        all_min = min([data[2].spectral_axis[0] for data in processed_data])
        all_max = max([data[2].spectral_axis[-1] for data in processed_data])
        step = processed_data[0][2].spectral_axis[1] - processed_data[0][2].spectral_axis[0]
        common_x = np.arange(all_min, all_max + step, step)

        # Подготовка данных
        wks_combined = wb_combined[0]
        wks_combined.lname = "Все_спектры"
        wks_combined.from_list(0, common_x, 'Raman_shift')

        sample_names = []
        sample_numbers = []
        id_ig_values = []
        i2d_ig_values = []
        id_ig_interpretations = []
        i2d_ig_interpretations = []

        for i, (name, _, processed_spec, peak_data, ratios, interpretations) in enumerate(processed_data):
            processed_x = processed_spec.spectral_axis
            processed_y = processed_spec.spectral_data[0]

            # Интерполяция на общую сетку
            interp_func = interp1d(processed_x, processed_y, kind='linear', bounds_error=False, fill_value=0)
            processed_y_interp = interp_func(common_x)

            wks_combined.from_list(i + 1, processed_y_interp, f'{name}')

            # Данные для отношений
            sample_names.append(name)
            sample_numbers.append(i + 1)

            id_ig_val = ratios.get('I_D/I_G', float('nan'))
            i2d_ig_val = ratios.get('I_2D/I_G', float('nan'))

            id_ig_values.append(id_ig_val)
            i2d_ig_values.append(i2d_ig_val)
            id_ig_interpretations.append(interpretations.get('I_D/I_G', 'N/A'))
            i2d_ig_interpretations.append(interpretations.get('I_2D/I_G', 'N/A'))

        # График всех спектров
        graph_combined = op.new_graph(template='line')
        graph_combined.lname = "Все обработанные спектры"

        colors = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(len(processed_data)):
            plot = graph_combined[0].add_plot(wks_combined, colx=0, coly=i + 1)
            plot.color = colors[i % len(colors)]
            plot.lname = processed_data[i][0]
            plot.line_width = 1.5

        graph_combined[0].rescale()
        graph_combined.set_str('xlabel', 'Волновое число (см⁻¹)')
        graph_combined.set_str('ylabel', 'Интенсивность')
        graph_combined.set_str('title', 'Все обработанные спектры')

        # Таблица отношений
        if sample_names:
            wks_ratios = wb_ratios[0]
            wks_ratios.lname = "Отношения"

            wks_ratios.from_list(0, sample_numbers, 'Номер_образца')
            wks_ratios.from_list(1, sample_names, 'Образец')
            wks_ratios.from_list(2, id_ig_values, 'I_D/I_G')
            wks_ratios.from_list(3, id_ig_interpretations, 'Интерпретация_I_D/I_G')
            wks_ratios.from_list(4, i2d_ig_values, 'I_2D/I_G')
            wks_ratios.from_list(5, i2d_ig_interpretations, 'Интерпретация_I_2D/I_G')

            # График I_D/I_G
            if any(not np.isnan(v) for v in id_ig_values):
                graph_id_ig = op.new_graph(template='scatter')
                graph_id_ig.lname = "I_D/I_G"
                plot_id_ig = graph_id_ig[0].add_plot(wks_ratios, colx=0, coly=2)
                plot_id_ig.color = 1
                plot_id_ig.size = 3
                graph_id_ig[0].rescale()
                graph_id_ig.set_str('xlabel', 'Номер образца')
                graph_id_ig.set_str('ylabel', 'I_D/I_G')
                graph_id_ig.set_str('title', 'Отношение I_D/I_G')
                print("Построен график I_D/I_G")

            # График I_2D/I_G
            if any(not np.isnan(v) for v in i2d_ig_values):
                graph_i2d_ig = op.new_graph(template='scatter')
                graph_i2d_ig.lname = "I_2D/I_G"
                plot_i2d_ig = graph_i2d_ig[0].add_plot(wks_ratios, colx=0, coly=4)
                plot_i2d_ig.color = 2
                plot_i2d_ig.size = 3
                graph_i2d_ig[0].rescale()
                graph_i2d_ig.set_str('xlabel', 'Номер образца')
                graph_i2d_ig.set_str('ylabel', 'I_2D/I_G')
                graph_i2d_ig.set_str('title', 'Отношение I_2D/I_G')
                print("Построен график I_2D/I_G")

        print(f"Созданы графики для {len(sample_names)} образцов")

    except Exception as e:
        print(f"Ошибка создания графиков в Origin: {e}")
        import traceback
        traceback.print_exc()


def generate_summary_report(processed_data, output_dir):
    """Генерация отчета"""
    report_path_txt = os.path.join(output_dir, "summary_report_voigt.txt")
    report_path_csv = os.path.join(output_dir, "summary_report_voigt.csv")

    summary_data = []
    for i, (name, _, _, peak_data, ratios, interpretations) in enumerate(processed_data):

        peak_details = []
        for peak in peak_data:
            peak_details.append(f"{peak['name']}({peak['position']:.0f}см⁻¹)")

        row = {
            'Номер': i + 1,
            'Образец': name,
            'Пики': '; '.join(peak_details),
            'I_D/I_G': ratios.get('I_D/I_G', 'N/A'),
            'Интерпретация_I_D/I_G': interpretations.get('I_D/I_G', 'N/A'),
            'La_nm': ratios.get('La_nm', 'N/A'),
            'La_примечание': ratios.get('La_note', 'N/A'),
            'I_2D/I_G': ratios.get('I_2D/I_G', 'N/A'),
            'Интерпретация_I_2D/I_G': interpretations.get('I_2D/I_G', 'N/A'),
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(report_path_txt, sep='\t', index=False, encoding='utf-8')
    df.to_csv(report_path_csv, index=False, encoding='utf-8')

    print(f"Отчет сохранен: {report_path_txt} и {report_path_csv}")
    return df


def main():
    """Основная функция"""
    print("АВТОМАТИЗИРОВАННАЯ ОБРАБОТКА RAMAN-СПЕКТРОВ")
    print("=" * 50)
    print(f"Вход: {INPUT_DIR}")
    print(f"Выход: {OUTPUT_DIR}")
    print(f"Origin: {'да' if USE_ORIGIN else 'нет (только файлы)'}")
    print(f"Лазер (для La): {LASER_WAVELENGTH_NM} нм")

    origin_enabled = USE_ORIGIN
    if origin_enabled:
        try:
            op.set_show(True)
            test_wb = op.new_book()
            test_wb.destroy()
            print("Origin доступен")
        except Exception as e:
            print(f"Origin недоступен ({e}), продолжаем без графиков Origin.")
            origin_enabled = False

    ensure_dirs()

    # Поиск файлов
    files = glob.glob(os.path.join(str(INPUT_DIR), "*.txt"))
    print(f"Найдено файлов: {len(files)}")

    if not files:
        print("Файлы не найдены")
        return

    # Обработка данных
    processed_data = []

    for file in files:
        try:
            name = os.path.basename(file).replace(".txt", "")
            print(f"\nОБРАБОТКА: {name}")

            raman_shift, intensity = load_spectrum_two_column(file)

            # Обработка спектра
            processed_spec = process_with_ramanspy_controlled(raman_shift, intensity)

            # Аппроксимация пиков
            original_spec = rp.Spectrum(intensity.reshape(1, -1), raman_shift)
            peak_data = voigt_peak_fitting(processed_spec)

            # Расчет отношений
            ratios, interpretations = calculate_ratios_from_voigt(peak_data)

            processed_data.append((name, original_spec, processed_spec, peak_data, ratios, interpretations))

            print(f"Успешно: {name}")

        except Exception as e:
            print(f"Ошибка: {e}")

    # Результаты
    if processed_data:
        print(f"\nСохранение обработанных спектров...")
        save_processed_spectra_files(processed_data, str(OUTPUT_DIR))

        print(f"\nСоздание графиков в Origin...")
        create_origin_plots(processed_data, str(OUTPUT_DIR), origin_enabled=origin_enabled)

        print(f"\nГенерация отчета...")
        generate_summary_report(processed_data, str(OUTPUT_DIR))

        print(f"\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        for i, (name, _, _, _, ratios, interpretations) in enumerate(processed_data):
            print(f"\n#{i + 1} {name}:")
            print(f"I_D/I_G = {ratios.get('I_D/I_G', 'N/A')} - {interpretations.get('I_D/I_G', 'N/A')}")
            if ratios.get("La_nm") is not None:
                print(f"La (оценка) = {ratios.get('La_nm')} нм")
            print(f"I_2D/I_G = {ratios.get('I_2D/I_G', 'N/A')} - {interpretations.get('I_2D/I_G', 'N/A')}")

    print(f"\nОбработка завершена")
    print(f"\nСозданы файлы:")
    print(f"- Очищенные спектры: *_processed.txt и *_processed.csv")
    print(f"- Сводный отчет: summary_report_voigt.txt и summary_report_voigt.csv")
    if origin_enabled:
        print(f"- Графики в Origin: сравнительный анализ спектров и отношений интенсивностей")


if __name__ == "__main__":
    main()
