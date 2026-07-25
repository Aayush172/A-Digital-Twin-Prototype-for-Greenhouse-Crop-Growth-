"""
Greenhouse Environment Physics Simulation.
Models diurnal variations in solar radiation, temperature, relative humidity, and CO2 concentration.
"""

import numpy as np
from datetime import datetime

class GreenhouseEnvironment:
    """Models physics-informed environmental dynamics inside a controlled greenhouse."""

    def __init__(self, base_temp: float = 22.0, temp_amplitude: float = 6.0,
                 base_humidity: float = 70.0, base_co2: float = 800.0,
                 max_par: float = 850.0):
        self.base_temp = base_temp
        self.temp_amplitude = temp_amplitude
        self.base_humidity = base_humidity
        self.base_co2 = base_co2
        self.max_par = max_par

    def compute_environmental_reading(self, dt: datetime, noise_level: float = 0.05, seed_rng: np.random.Generator = None) -> dict:
        """Computes environmental parameters at a given datetime."""
        if seed_rng is None:
            seed_rng = np.random.default_rng()

        hour = dt.hour + dt.minute / 60.0

        # Solar PAR (Light Intensity): zero at night, peak around solar noon (12:00-14:00)
        if 6.0 <= hour <= 18.0:
            solar_phase = np.pi * (hour - 6.0) / 12.0
            par_base = self.max_par * np.sin(solar_phase) ** 1.5
        else:
            par_base = 0.0

        # Temperature: solar heating leads peak by ~2 hours (peak ~ 14:00)
        temp_phase = 2.0 * np.pi * (hour - 8.0) / 24.0
        temp_base = self.base_temp + self.temp_amplitude * np.sin(temp_phase)

        # Humidity: inversely correlated with temperature
        humidity_base = self.base_humidity - (temp_base - self.base_temp) * 2.2

        # CO2 Concentration: photosynthesis depletes CO2 during daylight, enrichment keeps base
        if 6.0 <= hour <= 18.0:
            co2_depletion = 150.0 * np.sin(np.pi * (hour - 6.0) / 12.0)
            co2_base = self.base_co2 - co2_depletion
        else:
            co2_base = self.base_co2 + 50.0 # nocturnal respiration accumulation

        # Inject noise
        temp = float(temp_base + seed_rng.normal(0, noise_level * 1.5))
        humidity = float(np.clip(humidity_base + seed_rng.normal(0, noise_level * 3.0), 30.0, 95.0))
        co2 = float(np.clip(co2_base + seed_rng.normal(0, noise_level * 20.0), 350.0, 1500.0))
        light = float(np.clip(par_base + seed_rng.normal(0, noise_level * 25.0), 0.0, 1200.0))

        return {
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "co2": round(co2, 2),
            "light_intensity": round(light, 2)
        }
