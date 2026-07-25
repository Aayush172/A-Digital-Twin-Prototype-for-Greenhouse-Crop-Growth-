"""
Crop Growth Biology Model.
Calculates crop biomass accumulation and height dynamics using thermal time (growing degree days - GDD)
and non-linear stress functions for temperature, relative humidity, light, and CO2.
"""

import numpy as np
from utils.constants import (
    MAX_PLANT_HEIGHT_CM,
    OPTIMAL_TEMP_MIN,
    OPTIMAL_TEMP_MAX,
    OPTIMAL_HUMIDITY_MIN,
    OPTIMAL_HUMIDITY_MAX,
    OPTIMAL_CO2_MIN,
    OPTIMAL_CO2_MAX,
    OPTIMAL_PAR_MIN,
    OPTIMAL_PAR_MAX
)

class PlantGrowthModel:
    """Simulates physiological crop growth based on microclimate conditions."""

    def __init__(self, base_temp: float = 10.0, max_height: float = MAX_PLANT_HEIGHT_CM):
        self.base_temp = base_temp
        self.max_height = max_height
        self.cumulative_gdd = 0.0

    def reset(self):
        """Resets cumulative thermal time."""
        self.cumulative_gdd = 0.0

    def compute_temperature_factor(self, temp: float) -> float:
        """Cardio-thermal response function [0.0 - 1.0]."""
        if temp < self.base_temp or temp > 38.0:
            return 0.0
        elif OPTIMAL_TEMP_MIN <= temp <= OPTIMAL_TEMP_MAX:
            return 1.0
        elif temp < OPTIMAL_TEMP_MIN:
            return (temp - self.base_temp) / (OPTIMAL_TEMP_MIN - self.base_temp)
        else:
            return (38.0 - temp) / (38.0 - OPTIMAL_TEMP_MAX)

    def compute_light_factor(self, par_light: float) -> float:
        """Photosynthetic light response curve (Michaelis-Menten kinetics)."""
        if par_light <= 0:
            return 0.0
        half_saturation = 350.0
        return par_light / (par_light + half_saturation)

    def compute_co2_factor(self, co2: float) -> float:
        """CO2 fertilization growth response factor."""
        ambient_ref = 400.0
        if co2 <= 200.0:
            return 0.2
        # Moderate growth enhancement up to 1000 ppm
        return float(np.clip(1.0 + 0.35 * (co2 - ambient_ref) / 600.0, 0.5, 1.4))

    def compute_humidity_factor(self, humidity: float) -> float:
        """Vapour pressure deficit (VPD) proxy stress curve."""
        if OPTIMAL_HUMIDITY_MIN <= humidity <= OPTIMAL_HUMIDITY_MAX:
            return 1.0
        elif humidity < OPTIMAL_HUMIDITY_MIN:
            return float(np.clip(humidity / OPTIMAL_HUMIDITY_MIN, 0.3, 1.0))
        else:
            return float(np.clip((100.0 - humidity) / (100.0 - OPTIMAL_HUMIDITY_MAX), 0.3, 1.0))

    def compute_step_growth(self, temp: float, humidity: float, co2: float,
                            par_light: float, delta_days: float, current_height: float) -> float:
        """Computes plant height increment over delta_days."""
        # Thermal time increment (GDD)
        daily_temp_eff = max(0.0, temp - self.base_temp)
        self.cumulative_gdd += daily_temp_eff * delta_days

        # Environmental stress multipliers
        f_temp = self.compute_temperature_factor(temp)
        f_light = self.compute_light_factor(par_light)
        f_co2 = self.compute_co2_factor(co2)
        f_hum = self.compute_humidity_factor(humidity)

        combined_env_multiplier = f_temp * (0.6 * f_light + 0.4) * f_co2 * f_hum

        # Sigmoidal growth rate factor (logistic growth ceiling)
        height_ratio = current_height / self.max_height
        logistic_modifier = max(0.01, 1.0 - height_ratio)

        # Potential baseline growth rate: ~2.8 cm / day during peak vegetative phase
        base_rate = 2.8 * delta_days

        growth_increment = base_rate * combined_env_multiplier * logistic_modifier
        return float(max(0.0, growth_increment))
