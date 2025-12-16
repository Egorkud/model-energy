import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Кліматичні дані (Київ)
MONTHS = [
    'Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень',
    'Липень', 'Серпень', 'Вересень', 'Жовтень', 'Листопад', 'Грудень'
]
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
HOURS_IN_MONTH = DAYS_IN_MONTH * 24

# Середньомісячна зовнішня температура, °C
T_EXT = np.array([
    -3.5, -2.5, 2.0, 9.5, 15.5, 19.0,
    21.0, 20.0, 14.5, 8.0, 2.0, -1.5
])

# 2. Параметри приміщення (Ваш варіант)
T_INT_HEAT = 20.0  # Внутрішня задана температура опалення, °C
T_INT_COOL = 27.0  # Внутрішня задана температура охолодження, °C
VOLUME_BUILDING = 150.0  # Об'єм приміщення, м³
AREA_WALLS = 100.0  # Площа зовнішніх стін, м²
AREA_WINDOWS = 15.0  # Площа вікон, м²
AREA_ROOF = 50.0  # Площа даху/перекриття, м²
HEAT_CAPACITY = 1000.0  # Ефективна внутрішня теплоємність, Дж/(м²·К)

# 3. Базовий сценарій (U-коефіцієнти, Вт/(м²·К))
U_WALL_BASE = 1.20
U_WINDOW_BASE = 2.8
U_ROOF_BASE = 1.0

# 4. Покращений сценарій (Модернізація)
U_WALL_NEW = 0.25
U_WINDOW_NEW = 1.1
U_ROOF_NEW = 0.20
N_AIR_BASE = 0.5  # Кратність повітрообміну, 1/год (базовий)
N_AIR_NEW = 0.25  # Кратність повітрообміну, 1/год (покращений)
ETA_RECUP = 0.75  # Ефективність рекуператора

# 5. Внутрішні теплонадходження (Вт)
Q_INT_MONTHLY_W = 400 * np.ones(12)

# 6. Допоміжні константи
RHO_AIR = 1.20  # Густина повітря, кг/м³
C_AIR = 1000  # Питома теплоємність повітря, Дж/(кг·К)


def calculate_heating_losses(U_wall, U_window, U_roof, N_air, eta_recup=0.0):
    """
    Розраховує сумарний коефіцієнт тепловтрат (H_sum, Вт/К)
    через огороджувальні конструкції (H_tr) та вентиляцію (H_ve).
    """
    # Коефіцієнт теплопередачі через огороджувальні конструкції
    H_tr = (AREA_WALLS * U_wall +
            AREA_WINDOWS * U_window +
            AREA_ROOF * U_roof)

    # Коефіцієнт вентиляційних втрат
    H_ve = RHO_AIR * C_AIR * VOLUME_BUILDING * N_air / 3600 * (1 - eta_recup)

    H_sum = H_tr + H_ve
    return H_sum, H_tr, H_ve


def calculate_monthly_energy(H_sum, Q_int_W, T_int, T_ext_array, hours_in_month_array):
    """
    Розраховує щомісячну потребу в енергії на опалення (Qnd_H).
    Використовується спрощена стаціонарна модель.
    """
    T_int_array = np.full(12, T_int)

    # Щомісячні внутрішні надходження, Вт·год
    Q_int_monthly_Wh = Q_int_W * hours_in_month_array

    # Щомісячні тепловтрати (без надходжень), Вт·год
    Q_loss_Wh = H_sum * (T_int_array - T_ext_array) * hours_in_month_array

    # Потреба в опаленні (Qnd_H), Вт·год: тільки коли втрати > надходжень
    Qnd_H_Wh = np.maximum(0, Q_loss_Wh - Q_int_monthly_Wh)

    # Потреба в охолодженні (ігнорується в цій спрощеній моделі)
    Qnd_C_Wh = np.zeros(12)

    return Qnd_H_Wh, Qnd_C_Wh


def run_calculation():
    """
    Виконує розрахунок для базового та покращеного сценаріїв.
    """
    print(" 1. Розрахунок теплотехнічних коефіцієнтів")

    # БАЗОВИЙ СЦЕНАРІЙ
    H_sum_base, H_tr_base, H_ve_base = calculate_heating_losses(
        U_WALL_BASE, U_WINDOW_BASE, U_ROOF_BASE, N_AIR_BASE
    )

    # ПОКРАЩЕНИЙ СЦЕНАРІЙ
    H_sum_new, H_tr_new, H_ve_new = calculate_heating_losses(
        U_WALL_NEW, U_WINDOW_NEW, U_ROOF_NEW, N_AIR_NEW, ETA_RECUP
    )

    print(f"Базовий: H_sum = {H_sum_base:.2f} Вт/К")
    print(f"Покращений: H_sum = {H_sum_new:.2f} Вт/К")

    print("\n 2. Розрахунок щомісячної енергопотреби на опалення")

    Qnd_H_base, Qnd_C_base = calculate_monthly_energy(
        H_sum_base, Q_INT_MONTHLY_W, T_INT_HEAT, T_EXT, HOURS_IN_MONTH
    )

    Qnd_H_new, Qnd_C_new = calculate_monthly_energy(
        H_sum_new, Q_INT_MONTHLY_W, T_INT_HEAT, T_EXT, HOURS_IN_MONTH
    )

    # Конвертація в кВт·год
    Qnd_H_base_kwh = Qnd_H_base / 1000
    Qnd_H_new_kwh = Qnd_H_new / 1000

    # Формування результату
    results = pd.DataFrame({
        'Місяць': MONTHS,
        'T_зовн, °C': T_EXT,
        'H_базовий, Вт/К': H_sum_base,
        'H_покращений, Вт/К': H_sum_new,
        'Потр_опал_БАЗА, кВт·год': Qnd_H_base_kwh,
        'Потр_опал_НОВИЙ, кВт·год': Qnd_H_new_kwh,
    })

    # Додавання рядка "РАЗОМ"
    total_row = pd.DataFrame([{
        'Місяць': 'РАЗОМ',
        'T_зовн, °C': np.nan,
        'H_базовий, Вт/К': np.nan,
        'H_покращений, Вт/К': np.nan,
        'Потр_опал_БАЗА, кВт·год': Qnd_H_base_kwh.sum(),
        'Потр_опал_НОВИЙ, кВт·год': Qnd_H_new_kwh.sum(),
    }])
    results = pd.concat([results, total_row], ignore_index=True)

    return results, H_sum_base, H_sum_new


def display_and_save_results(df, H_base, H_new):
    """
    Виводить результати, зберігає у файл Excel та будує графік.
    """
    file_name = "results.xlsx"

    # Виведення у консоль
    print("\n" + "=" * 80)
    print("\t\tЗВІТ ПРО ЕНЕРГОПОТРЕБУ НА ОПАЛЕННЯ")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.2f"))

    # Виведення підсумків
    total_base = df.loc[df['Місяць'] == 'РАЗОМ', 'Потр_опал_БАЗА, кВт·год'].values[0]
    total_new = df.loc[df['Місяць'] == 'РАЗОМ', 'Потр_опал_НОВИЙ, кВт·год'].values[0]
    reduction_kwh = total_base - total_new
    reduction_perc = (reduction_kwh / total_base) * 100

    print("\n" + "=" * 80)
    print(f"Річна потреба (БАЗА): {total_base:.2f} кВт·год")
    print(f"Річна потреба (НОВА): {total_new:.2f} кВт·год")
    print(f"Економія енергії: {reduction_kwh:.2f} кВт·год ({reduction_perc:.1f}%)")
    print("=" * 80)
    print(f"Сумарний коефіцієнт тепловтрат (База -> Новий): {H_base:.2f} Вт/К -> {H_new:.2f} Вт/К")

    # Збереження в Excel
    try:
        df.to_excel(file_name, index=False, float_format="%.2f")
        print(f"\nРезультати збережено у файлі: '{file_name}'")
    except Exception as e:
        print(f"\nПомилка збереження у Excel: {e}")

    # Побудова графіка
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    months_labels = df['Місяць'][:-1]
    base_data = df['Потр_опал_БАЗА, кВт·год'][:-1]
    new_data = df['Потр_опал_НОВИЙ, кВт·год'][:-1]
    x_indices = np.arange(len(months_labels))

    plt.bar(x_indices - 0.2, base_data, 0.4, label='Базовий сценарій', color='#e74c3c')
    plt.bar(x_indices + 0.2, new_data, 0.4, label='Покращений сценарій (Модернізація)', color='#2ecc71')

    plt.title('Порівняння щомісячної потреби в енергії на опалення', fontsize=16)
    plt.xlabel('Місяць', fontsize=12)
    plt.ylabel('Потреба в енергії, кВт·год', fontsize=12)
    plt.xticks(x_indices, months_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_df, H_base, H_new = run_calculation()
    display_and_save_results(results_df, H_base, H_new)
