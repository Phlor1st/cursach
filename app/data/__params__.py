# --- КОМБИНАЦИИ ЭЛЕМЕНТОВ (ВОСПРОИЗВОДИМЫЙ ПОРЯДОК) ---
# ПОРЯДОК ВАЖЕН! Новые комбинации добавлять только в конец.
ALL_ELEMENT_COMBINATIONS = [
    "C-Al",
    "C-Co",
    "C-Ta",
]

# --- АВТОНУМЕРАЦИЯ КОМБИНАЦИЙ ---
COMBINATION_TO_INT = {name: i for i, name in enumerate(sorted(list(ALL_ELEMENT_COMBINATIONS)))}
N_COMBINATIONS = len(COMBINATION_TO_INT)
