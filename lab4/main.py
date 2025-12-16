import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# КОНФІГУРАЦІЯ ТА КОНСТАНТИ
# ==========================================
# Ваги критеріїв для інтегральної оцінки (сума має бути 1.0)
WEIGHTS = {
    'efficiency': 0.2,  # Енергетична ефективність
    'reliability': 0.1,  # Надійність/сервіс
    'cost': 0.1,  # Вартість
    'ecology': 0.4,  # Екологічність/шум
    'compatibility': 0.2  # Сумісність із місцевістю
}

# Параметри місцевості (Alpha - коефіцієнт шорсткості)
TERRAIN_PARAMS = {
    'field': {'alpha': 0.15, 'name': 'Відкрите поле'},
    'coast': {'alpha': 0.15, 'name': 'Морське узбережжя'},
    'mountain': {'alpha': 0.25, 'name': 'Гірська місцевість'},
    'forest': {'alpha': 0.25, 'name': 'Ліс / пагорби'},
    'city': {'alpha': 0.32, 'name': 'Міська забудова'}
}


# ==========================================
# КЛАСИ
# ==========================================

class TurbineType:
    def __init__(self, name_key, name_display, cp, v_cut_in, v_rated, v_cut_out, area_type, kv=1.0, dh_ratio=None,
                 scores=None):
        self.key = name_key
        self.name = name_display
        self.cp = cp
        self.v_cut_in = v_cut_in
        self.v_rated = v_rated
        self.v_cut_out = v_cut_out
        self.area_type = area_type  # 'axial', 'vertical', 'savonius'
        self.kv = kv  # Коефіцієнт прискорення (для DAWT)
        self.dh_ratio = dh_ratio  # Відношення D/H для вертикальних
        self.scores = scores if scores else {}  # Матриця оцінок для різних місцевостей

    def get_score(self, terrain_key):
        """Отримує словник оцінок (q) для конкретної місцевості"""
        return self.scores.get(terrain_key, {
            'efficiency': 0.5, 'reliability': 0.5, 'cost': 0.5, 'ecology': 0.5, 'compatibility': 0.5
        })

    def calculate_area(self, p_plan_kw, p_kin_avg):
        """Розрахунок площі: A = P_plan / (P_kin * Cp)"""
        # P_plan у Вт, P_kin_avg у Вт/м2
        if p_kin_avg <= 0: return 0
        return (p_plan_kw * 1000) / (p_kin_avg * self.cp)

    def calculate_dimensions(self, area):
        """Розрахунок D та H"""
        if self.area_type == 'axial':  # A = pi * D^2 / 4
            d = math.sqrt((4 * area) / math.pi)
            return d, d  # H умовно дорівнює D для візуалізації

        elif self.area_type == 'vertical':  # A = D * H
            # Darrieus: D/H = 0.5 => H = 2D => A = D * 2D = 2D^2 => D = sqrt(A/2)
            ratio = self.dh_ratio if self.dh_ratio else 0.5
            # A = D * (D / ratio) => A = D^2 / ratio => D = sqrt(A * ratio)
            # Чекайте, D/H = ratio => D = H * ratio => H = D / ratio.
            # A = D * H = D * (D/ratio) = D^2 / ratio.
            # Для Darrieus (D/H=0.5): A = D * 2D = 2D^2. D = sqrt(A/2).
            # Перевірка: D/H = D/2D = 0.5. Правильно.

            # Використовуємо фіксовану логіку з методички:
            if self.key == 'darrieus':  # D/H = 0.5 -> H = 2D
                d = math.sqrt(area / 2.0)
                h = 2.0 * d
            elif self.key == 'savonius':  # D/H = 1.0 -> H = D, A = 0.8 * D * H = 0.8 D^2
                d = math.sqrt(area / 0.8)
                h = d
            else:  # Загальний випадок vertical
                d = math.sqrt(area)
                h = d
            return d, h
        return 0, 0


# База даних турбін з оцінками (Нормовані 0..1)
TURBINES = [
    TurbineType('hawt', 'Осьова (HAWT)', 0.45, 3.0, 12.0, 25.0, 'axial',
                scores={
                    'field': {'efficiency': 1.0, 'reliability': 0.8, 'cost': 0.9, 'ecology': 0.7, 'compatibility': 0.9},
                    'coast': {'efficiency': 1.0, 'reliability': 0.6, 'cost': 0.8, 'ecology': 0.8, 'compatibility': 1.0},
                    'city': {'efficiency': 0.4, 'reliability': 0.8, 'cost': 0.9, 'ecology': 0.2,
                             'compatibility': 0.1}}),

    TurbineType('dawt', 'Дифузорна (DAWT)', 0.50, 2.5, 10.0, 22.0, 'axial', kv=1.4,
                scores={
                    'field': {'efficiency': 0.9, 'reliability': 0.7, 'cost': 0.6, 'ecology': 0.8, 'compatibility': 0.8},
                    'coast': {'efficiency': 0.9, 'reliability': 0.5, 'cost': 0.5, 'ecology': 0.8, 'compatibility': 0.9},
                    'city': {'efficiency': 0.6, 'reliability': 0.7, 'cost': 0.6, 'ecology': 0.5,
                             'compatibility': 0.4}}),

    TurbineType('darrieus', 'Вертикальна Darrieus', 0.35, 3.5, 11.0, 25.0, 'vertical', dh_ratio=0.5,
                scores={
                    'city': {'efficiency': 0.7, 'reliability': 0.6, 'cost': 0.7, 'ecology': 0.9, 'compatibility': 0.8},
                    'mountain': {'efficiency': 0.8, 'reliability': 0.7, 'cost': 0.7, 'ecology': 0.9,
                                 'compatibility': 0.9},
                    'field': {'efficiency': 0.6, 'reliability': 0.6, 'cost': 0.8, 'ecology': 0.9,
                              'compatibility': 0.7}}),

    TurbineType('savonius', 'Savonius', 0.20, 1.5, 7.0, 18.0, 'vertical', dh_ratio=1.0,
                scores={
                    'city': {'efficiency': 0.3, 'reliability': 0.9, 'cost': 0.9, 'ecology': 1.0, 'compatibility': 1.0},
                    'mountain': {'efficiency': 0.3, 'reliability': 0.9, 'cost': 0.9, 'ecology': 1.0,
                                 'compatibility': 0.9}}),

    TurbineType('helical', 'Спіральна (Helical)', 0.35, 2.5, 10.0, 24.0, 'axial',  # Area formula same as axial usually
                scores={
                    'city': {'efficiency': 0.8, 'reliability': 0.9, 'cost': 0.7, 'ecology': 1.0, 'compatibility': 1.0},
                    'mountain': {'efficiency': 0.8, 'reliability': 0.9, 'cost': 0.7, 'ecology': 1.0,
                                 'compatibility': 0.9}})
]


# ==========================================
# ФУНКЦІЇ РОЗРАХУНКУ
# ==========================================

def generate_mock_pvgis_data(lat, lon, terrain):
    """
    Генерує реалістичні погодинні дані (8760 рядків) для емуляції PVGIS CSV.
    Використовує сезонні коливання та добові цикли.
    """
    np.random.seed(42)
    hours = 8760

    # Базова швидкість залежить від місцевості
    base_speed = 5.0 if terrain in ['field', 'coast'] else 3.5
    if terrain == 'city': base_speed = 3.0

    # Часові індекси
    time_index = pd.date_range(start='2024-01-01', periods=hours, freq='H')

    # Моделювання температури (сезонність)
    # Зима холодна, літо тепле. Максимум в липні (близько 5000-ї години)
    day_of_year = time_index.dayofyear
    hour_of_day = time_index.hour

    temp_seasonal = -5 + 25 * np.sin((day_of_year - 100) * 2 * np.pi / 365) ** 2
    temp_daily = 5 * np.sin((hour_of_day - 4) * 2 * np.pi / 24)
    temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, hours)

    # Моделювання вітру (Вейбулл розподіл + сезонність)
    # Вітер сильніший взимку та вдень
    wind_seasonal = 1 + 0.2 * np.cos((day_of_year) * 2 * np.pi / 365)
    wind_daily = 1 + 0.3 * np.sin((hour_of_day - 9) * 2 * np.pi / 24)
    # Базовий шум (Rayleigh/Weibull-like)
    wind_noise = np.random.weibull(2.0, hours) * base_speed

    wind_speed_10m = wind_noise * wind_seasonal * wind_daily

    # Тиск (близько 101.3 кПа з шумом)
    pressure = 101.3 + np.random.normal(0, 1, hours)

    df = pd.DataFrame({
        'time': time_index,
        'T2m': temperature,  # C
        'WS10m': wind_speed_10m,  # m/s
        'SP': pressure * 1000  # Pa (convert from kPa if logic requires, standard meteo is often Pa or hPa)
    })

    # У методичці формула P в кПа, тут генеруємо Па для стандарту, конвертуємо в функції
    return df


def calculate_air_density(temp_c, pressure_pa):
    """
    Розрахунок густини повітря: rho = 3.488 * P(kPa) / T(K)
    """
    pressure_kpa = pressure_pa / 1000.0
    temp_k = temp_c + 273.15
    return (3.488 * pressure_kpa) / temp_k


def extrapolate_wind_speed(v_ref, h_ref, h_target, alpha):
    """
    Екстраполяція швидкості вітру за законом Хеллмана
    """
    if v_ref < 0: return 0
    return v_ref * (h_target / h_ref) ** alpha


def select_best_turbine(terrain_key):
    """
    Вибір турбіни на основі інтегральної оцінки
    """
    best_turbine = None
    best_score = -1.0
    print(f"\n--- Інтегральна оцінка для місцевості: {TERRAIN_PARAMS[terrain_key]['name']} ---")
    print(f"{'Тип турбіни':<25} | {'Оцінка':<10} | {'Деталі'}")
    print("-" * 60)

    for t in TURBINES:
        scores = t.get_score(terrain_key)
        # Інтегральна сума: Sum(Wi * qi)
        total_q = sum(WEIGHTS[crit] * scores[crit] for crit in WEIGHTS)

        print(f"{t.name:<25} | {total_q:.3f}      | Cp={t.cp}, Vin={t.v_cut_in}")

        if total_q > best_score:
            best_score = total_q
            best_turbine = t

    print("-" * 60)
    print(f"ПЕРЕМОЖЕЦЬ: {best_turbine.name}\n")
    return best_turbine


def calculate_turbine_power(v_wind, turbine, rho, area, p_nom_kw):
    """
    Розрахунок миттєвої потужності турбіни (кВт) за узагальненою кривою.
    Враховує Kv для дифузорних турбін.
    """
    # Ефективна швидкість (для DAWT Kv > 1, для інших Kv = 1)
    v_eff = v_wind * turbine.kv

    if v_eff < turbine.v_cut_in or v_eff > turbine.v_cut_out:
        return 0.0

    if v_eff >= turbine.v_rated:
        return p_nom_kw  # Номінальна потужність

    # На ділянці від cut-in до rated: кубічна залежність або формула P = 0.5*rho*A*v^3*Cp
    # Але оскільки ми маємо досягти P_nom при V_rated, краще інтерполювати
    # P(v) = P_nom * ((v^3 - vin^3) / (vrated^3 - vin^3)) - спрощена модель з методички
    # Або фізична модель:
    power_w = 0.5 * rho * area * (v_eff ** 3) * turbine.cp
    power_kw = power_w / 1000.0

    # Обмежуємо номіналом, якщо раптом фізична формула дасть більше (через високе rho)
    return min(power_kw, p_nom_kw)


# ==========================================
# ГОЛОВНА ЛОГІКА
# ==========================================

def main():
    # 1. Вхідні дані
    print("=== ЛАБОРАТОРНА РОБОТА №4: ВИБІР ВЕС ===")

    # Можна замінити на input()
    city_name = "Київ"
    terrain_input = "city"  # 'city', 'field', 'coast', 'forest', 'mountain'
    mast_height = 15.0  # метрів
    p_plan_kw = 3.0  # кВт (завдання обмежує до 5)

    if p_plan_kw > 5:
        print("Попередження: Рекомендована потужність до 5 кВт.")

    # 2. Отримання метеоданих
    print(f"Генерація метеоданих для: {city_name}, місцевість: {terrain_input}...")
    df = generate_mock_pvgis_data(50.45, 30.52, terrain_input)

    # 3. Розрахунок параметрів вітру на висоті ротора
    alpha = TERRAIN_PARAMS[terrain_input]['alpha']

    # Додаємо стовпчики в DataFrame
    df['rho'] = df.apply(lambda row: calculate_air_density(row['T2m'], row['SP']), axis=1)
    df['WS_hub'] = df['WS10m'].apply(lambda v: extrapolate_wind_speed(v, 10.0, mast_height, alpha))

    # Кінетична потужність потоку на 1 м2: P = 0.5 * rho * 1 * v^3
    df['P_kin_m2'] = 0.5 * df['rho'] * (df['WS_hub'] ** 3)

    avg_p_kin = df['P_kin_m2'].mean()
    print(f"Середньорічна кінетична енергія потоку (на {mast_height}м): {avg_p_kin:.2f} Вт/м²")

    # 4. Вибір турбіни
    turbine = select_best_turbine(terrain_input)

    # 5. Розрахунок геометрії
    # A = P_plan / (P_kin_avg * Cp)
    area = turbine.calculate_area(p_plan_kw, avg_p_kin)
    diameter, height = turbine.calculate_dimensions(area)

    print(f"--- Геометричні параметри ---")
    print(f"Необхідна площа ротора: {area:.2f} м²")
    print(f"Діаметр (D): {diameter:.2f} м")
    if height != diameter:
        print(f"Висота (H): {height:.2f} м")

    # 6. Розрахунок генерації (Погодинно)
    df['Power_kW'] = df.apply(
        lambda row: calculate_turbine_power(row['WS_hub'], turbine, row['rho'], area, p_plan_kw),
        axis=1
    )

    # AEP (Annual Energy Production)
    # Оскільки дані погодинні, просто сумуємо потужність (кВт * 1 год)
    aep = df['Power_kW'].sum()
    capacity_factor = (aep / (p_plan_kw * 8760)) * 100

    print(f"\n--- Енергетичні показники ---")
    print(f"Річна генерація (AEP): {aep:.2f} кВт·год")
    print(f"Коефіцієнт використання встановленої потужності (КВВП): {capacity_factor:.2f}%")

    # 7. Аналіз та звіти

    # a) Помісячна генерація
    monthly_gen = df.groupby(df['time'].dt.month_name())['Power_kW'].sum().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])

    print("\nПомісячна генерація (кВт·год):")
    print(monthly_gen)

    # b) Сортування за швидкістю вітру (0-25 м/с)
    bins = range(0, 27)
    df['Speed_Bin'] = pd.cut(df['WS_hub'], bins=bins, right=False)
    speed_analysis = df.groupby('Speed_Bin', observed=False)['Power_kW'].agg(['mean', 'sum', 'count'])

    # 8. Графіки
    plt.figure(figsize=(15, 10))

    # Графік 1: Крива потужності (Теоретична vs Реальна середня)
    plt.subplot(2, 2, 1)
    v_range = np.linspace(0, 25, 100)
    # Беремо стандартну rho для кривої
    std_rho = 1.225
    p_curve = [calculate_turbine_power(v, turbine, std_rho, area, p_plan_kw) for v in v_range]
    plt.plot(v_range, p_curve, 'r-', label='Теоретична крива')
    # Додаємо точки реальної генерації (середнє по бінам)
    bin_centers = [b.mid for b in speed_analysis.index]
    plt.scatter(bin_centers, speed_analysis['mean'], alpha=0.6, label='Фактична середня')
    plt.title(f'Крива потужності ({turbine.name})')
    plt.xlabel('Швидкість вітру (м/с)')
    plt.ylabel('Потужність (кВт)')
    plt.grid(True)
    plt.legend()

    # Графік 2: Помісячна генерація
    plt.subplot(2, 2, 2)
    monthly_gen.plot(kind='bar', color='skyblue')
    plt.title('Помісячна генерація енергії')
    plt.ylabel('кВт·год')
    plt.grid(axis='y')

    # Графік 3: Гістограма швидкостей вітру
    plt.subplot(2, 2, 3)
    plt.hist(df['WS_hub'], bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(turbine.v_cut_in, color='r', linestyle='--', label='Cut-in')
    plt.axvline(turbine.v_rated, color='orange', linestyle='--', label='Rated')
    plt.title('Розподіл швидкості вітру на висоті ротора')
    plt.xlabel('Швидкість (м/с)')
    plt.ylabel('Годин на рік')
    plt.legend()

    # Графік 4: Погодинна генерація для одного дня (наприклад, 15 січня)
    plt.subplot(2, 2, 4)
    day_data = df[df['time'].dt.date == pd.to_datetime('2024-01-15').date()]
    plt.plot(day_data['time'].dt.hour, day_data['Power_kW'], marker='o')
    plt.title('Добовий графік (15 січня)')
    plt.xlabel('Година')
    plt.ylabel('Потужність (кВт)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()