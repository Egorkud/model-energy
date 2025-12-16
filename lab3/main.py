from math import sin, cos, acos, asin, radians, degrees

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# КОНФІГУРАЦІЯ ТА ВХІДНІ ДАНІ
# ==========================================

# Координати (Київ)
LATITUDE = 50.45  # Широта
LONGITUDE = 30.52  # Довгота

# Параметри СЕС
PANEL_POWER_KW = 1.0  # Встановлена потужність (кВт)
SYSTEM_EFFICIENCY = 0.85  # ККД системи (втрати в інверторі, кабелях, бруд тощо)

# Економічні параметри (з варіанту Мовчан/Колісніченко)
TARIFF_UAH = 6.0  # Тариф для населення, грн/кВт*год
COST_PER_KW = 15000  # Вартість 1 кВт "під ключ" (панелі + інвертор + монтаж), грн

# Дні для моделювання сезонів (день року)
# 21 березня (80), 21 червня (172), 21 вересня (264), 21 грудня (355)
SEASON_DAYS = {
    'Весна (21 березня)': 80,
    'Літо (21 червня)': 172,
    'Осінь (21 вересня)': 264,
    'Зима (21 грудня)': 355
}


# ==========================================
# МАТЕМАТИЧНА МОДЕЛЬ СОНЯЧНОЇ РАДІАЦІЇ
# ==========================================

class SolarModel:
    def __init__(self, lat, lon):
        self.lat_rad = radians(lat)
        self.lon = lon

    def calculate_position(self, day_of_year, hour):
        """
        Розраховує положення сонця (висота, азимут) для заданого дня та години.
        hour - локальний час (десятковий, наприклад 12.5 = 12:30)
        """
        # Схилення Сонця (declination), delta
        # Формула Купера
        delta = 23.45 * sin(radians(360 / 365 * (284 + day_of_year)))
        delta_rad = radians(delta)

        # Кутовий час (hour angle), omega
        # 12:00 сонячного часу = 0 градусів. Кожна година = 15 градусів.
        # Спрощення: вважаємо локальний час близьким до сонячного для навчальної моделі
        omega = 15 * (hour - 12)
        omega_rad = radians(omega)

        # Висота Сонця (Solar Elevation/Altitude), alpha
        sin_alpha = (sin(self.lat_rad) * sin(delta_rad) +
                     cos(self.lat_rad) * cos(delta_rad) * cos(omega_rad))

        alpha_rad = asin(max(-1, min(1, sin_alpha)))  # обмеження для уникнення помилок
        alpha_deg = degrees(alpha_rad)

        # Азимут Сонця (Solar Azimuth), phi
        # 0 = Північ, 90 = Схід, 180 = Південь, 270 = Захід (в астрономії часто Південь=0, тут використаємо геодезичну Пн=0)

        # Захист від ділення на нуль при зеніті
        try:
            cos_phi = (sin_alpha * sin(self.lat_rad) - sin(delta_rad)) / (cos(alpha_rad) * cos(self.lat_rad))
            cos_phi = max(-1, min(1, cos_phi))
            phi_rad = acos(cos_phi)
            phi_deg = degrees(phi_rad)

            # Коригування азимуту в залежності від часу доби
            if hour > 12:
                phi_deg = 360 - phi_deg
        except:
            phi_deg = 180  # Умовно південь

        return alpha_deg, phi_deg

    def calculate_irradiance(self, day_of_year, hour, panel_tilt, panel_azimuth, use_tracker=False):
        """
        Розраховує потужність випромінювання на панель (Вт/м2).
        panel_tilt: кут нахилу панелі до горизонту (0 - горизонтально, 90 - вертикально)
        panel_azimuth: азимут панелі (180 - південь)
        """
        alpha, sun_azimuth = self.calculate_position(day_of_year, hour)

        if alpha <= 0:
            return 0.0  # Сонце за горизонтом

        alpha_rad = radians(alpha)

        # Імітація прозорості атмосфери (Clear Sky Model)
        # Сонячна постійна
        I_sc = 1367
        # Повітряна маса (Air Mass)
        AM = 1 / (sin(alpha_rad) + 0.00001)  # +epsilon щоб не ділити на 0
        # Екстраполяція інтенсивності прямого випромінювання (DNI)
        # Проста модель: I_dir = I_sc * 0.7^(AM^0.678)
        I_dni = I_sc * (0.7 ** (AM ** 0.678))

        # Дифузне випромінювання (розсіяне) - спрощено 10% від прямого
        I_diff = I_dni * 0.1

        if use_tracker:
            # Трекер завжди тримає панель перпендикулярно до сонця
            cos_theta = 1
        else:
            # Кут падіння променів на фіксовану панель (Incidence Angle), theta
            beta_rad = radians(panel_tilt)
            gamma_rad = radians(panel_azimuth)  # азимут панелі
            phi_sun_rad = radians(sun_azimuth)  # азимут сонця

            # Формула кута падіння
            cos_theta = (sin(alpha_rad) * cos(beta_rad) +
                         cos(alpha_rad) * sin(beta_rad) * cos(phi_sun_rad - gamma_rad))

            # Якщо сонце світить "в спину" панелі
            if cos_theta < 0:
                cos_theta = 0

        # Сумарна радіація на похилу поверхню
        # Global Tilted Irradiance (GTI) = Direct * cos(theta) + Diffuse
        I_total = I_dni * cos_theta + I_diff

        return max(0, I_total)


# ==========================================
# ЛОГІКА МОДЕЛЮВАННЯ
# ==========================================

model = SolarModel(LATITUDE, LONGITUDE)


def simulate_day_generation(day, tilt, azimuth, use_tracker=False):
    """
    Симулює один день. Повертає список годин, потужностей (Вт) та сумарну енергію (кВт*год).
    """
    hours = np.arange(0, 24, 10 / 60)  # Крок 10 хвилин
    powers = []

    for h in hours:
        irradiance_w_m2 = model.calculate_irradiance(day, h, tilt, azimuth, use_tracker)
        # Потужність = Інсоляція * Площа * ККД панелі * ККД системи
        # Для спрощення: маємо 1 кВт панелей. 
        # Стандартні умови (STC) - це 1000 Вт/м2. Тому P_out = P_nom * (Irr / 1000)

        power_output_kw = PANEL_POWER_KW * (irradiance_w_m2 / 1000) * SYSTEM_EFFICIENCY
        powers.append(power_output_kw)

    # Інтегрування для отримання енергії (кВт*год)
    # Сума потужностей * крок часу (10 хв = 1/6 години)
    daily_energy_kwh = sum(powers) * (10 / 60)

    return hours, powers, daily_energy_kwh


def simulate_year_monthly(tilt, azimuth, use_tracker=False):
    """
    Симулює рік, повертає словник {місяць: енергія_кВт_год} та річну суму.
    Беремо 15-те число кожного місяця як середнє.
    """
    monthly_energy = []
    total_year = 0
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Середні дні місяців (15 січня = 15 день, 15 лютого = 46 день і т.д.)
    mid_month_days = [15, 46, 75, 105, 135, 166, 196, 227, 257, 287, 318, 348]

    for i, day in enumerate(mid_month_days):
        _, _, daily_kwh = simulate_day_generation(day, tilt, azimuth, use_tracker)
        month_kwh = daily_kwh * days_in_month[i]
        monthly_energy.append(month_kwh)
        total_year += month_kwh

    return monthly_energy, total_year


# ==========================================
# ВИКОНАННЯ ЗАВДАНЬ
# ==========================================

print(f"--- ЗАПУСК МОДЕЛЮВАННЯ СЕС ({LATITUDE}°, {LONGITUDE}°) ---")
print(f"Потужність: {PANEL_POWER_KW} кВт, ККД системи: {SYSTEM_EFFICIENCY * 100}%")

# --- ЗАВДАННЯ 1.1: Добова генерація по сезонах ---
plt.figure(figsize=(10, 6))
results_1_1 = {}

print("\n[1.1] Добова генерація (Оптимальний кут ~35°, Південь):")
for season, day in SEASON_DAYS.items():
    hours, powers, total = simulate_day_generation(day, tilt=35, azimuth=180)
    results_1_1[season] = total
    plt.plot(hours, powers, label=f"{season} (E={total:.2f} кВт·год)")
    print(f"  {season}: {total:.2f} кВт·год")

plt.title("Добова генерація СЕС (1 кВт) у різні пори року")
plt.xlabel("Година доби")
plt.ylabel("Потужність генерації (кВт)")
plt.xticks(range(0, 25, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# Графік буде показано в кінці або збережено

# --- ЗАВДАННЯ 1.2 & 1.3: Вплив кута нахилу та азимуту ---
print("\n[1.2-1.3] Аналіз річної генерації для різних конфігурацій:")

scenarios = [
    {"name": "Південь (180°), Нахил 30°", "tilt": 30, "az": 180},
    {"name": "Південь (180°), Нахил 45°", "tilt": 45, "az": 180},
    {"name": "Південь (180°), Нахил 60°", "tilt": 60, "az": 180},
    {"name": "Схід (90°), Нахил 30°", "tilt": 30, "az": 90},
    {"name": "Захід (270°), Нахил 30°", "tilt": 30, "az": 270},
    {"name": "Горизонтально (0°)", "tilt": 0, "az": 180},  # Азимут не важливий
]

tracker_monthly, tracker_year = simulate_year_monthly(0, 0, use_tracker=True)
scenarios_results = []

plt.figure(figsize=(12, 6))
months_names = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер', 'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']

# Розрахунок звичайних сценаріїв
best_static_gen = 0
best_static_name = ""

for sc in scenarios:
    m_res, y_res = simulate_year_monthly(sc["tilt"], sc["az"])
    scenarios_results.append((sc["name"], y_res))
    plt.plot(months_names, m_res, marker='o', label=f'{sc["name"]} ({y_res:.0f} кВт·год/рік)')

    if y_res > best_static_gen:
        best_static_gen = y_res
        best_static_name = sc["name"]

# Додаємо трекер на графік
plt.plot(months_names, tracker_monthly, marker='s', linewidth=3, linestyle='--', color='red',
         label=f'Двоосьовий Трекер ({tracker_year:.0f} кВт·год/рік)')

plt.title("Помісячна генерація для різних орієнтацій та трекера")
plt.ylabel("Енергія (кВт·год / міс)")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# --- ВИСНОВКИ В КОНСОЛЬ ---
print(f"{'Конфігурація':<35} | {'Річна генерація (кВт·год)':<25}")
print("-" * 65)
for name, val in scenarios_results:
    print(f"{name:<35} | {val:.2f}")
print(f"{'Двоосьовий Трекер':<35} | {tracker_year:.2f}")

tracker_increase = ((tracker_year - best_static_gen) / best_static_gen) * 100
print(f"\nПриріст від трекера порівняно з найкращим статичним варіантом ({best_static_name}): +{tracker_increase:.1f}%")

# --- ДОДАТКОВЕ ЗАВДАННЯ: ЕКОНОМІКА ---
print("\n[Додатково] Техніко-економічний розрахунок (для власного споживання)")

# Використовуємо найкращий статичний варіант (для реалістичності, бо трекер дорогий)
annual_revenue = best_static_gen * TARIFF_UAH
payback_period = COST_PER_KW / annual_revenue

print(f"Вартість встановлення 1 кВт: {COST_PER_KW} грн")
print(f"Тариф електроенергії: {TARIFF_UAH} грн/кВт·год")
print(f"Річна економія (кращий статичний): {annual_revenue:.2f} грн")
print(f"Термін окупності: {payback_period:.1f} років")

plt.show()
