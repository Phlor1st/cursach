"""
Количественные оценки по литературным моделям для спектров графена.

Основная связь размера кристаллита La с I_D/I_G — эмпирическая формула
Cançado et al. (см. обзоры по Raman графена): для нанокристаллического
графита/графена часто используют масштабирование с длиной волны лазера.

La [нм] ≈ (560 / E_L^4) * (I_G / I_D), где E_L — энергия фотона [эВ],
I_G/I_D = 1 / (I_D/I_G). При малых I_D/I_G и сильной дисперсии модель
нужно интерпретировать осторожно; в пояснительной записке указать
источник и допущения.
"""


def photon_energy_ev(wavelength_nm: float) -> float:
    """E [эВ] ≈ 1239.84 / λ [нм]."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")
    return 1239.84193 / wavelength_nm


def crystallite_size_la_nm(id_over_ig: float, laser_wavelength_nm: float = 532.0):
    """
    Оценка размера кристаллита La [нм] по отношению интенсивностей I_D/I_G.

    Используется вид La = (560 / E_L^4) * (I_G / I_D) с E_L в эВ
    (распространённая форма в литературе по Raman углерода).

    Возвращает None, если отношение неположительно или неопределено.
    """
    if id_over_ig is None or id_over_ig <= 0:
        return None
    ig_over_id = 1.0 / id_over_ig
    e_l = photon_energy_ev(laser_wavelength_nm)
    la_nm = (560.0 / (e_l**4)) * ig_over_id
    return float(la_nm)


def format_physics_summary(id_over_ig, laser_nm: float) -> dict:
    """Словарь для отчёта: La и подпись."""
    if id_over_ig is None:
        return {"La_nm": None, "La_note": "I_D/I_G не определено"}
    la = crystallite_size_la_nm(id_over_ig, laser_nm)
    if la is None:
        return {"La_nm": None, "La_note": "некорректное I_D/I_G"}
    return {
        "La_nm": round(la, 2),
        "La_note": f"оценка La ≈ {la:.2f} нм (модель Cançado, λ={laser_nm:.0f} нм)",
    }
