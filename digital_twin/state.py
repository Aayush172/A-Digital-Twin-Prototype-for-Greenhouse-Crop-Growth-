"""
Digital Twin State representation.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from typing import Dict, Any

@dataclass
class DigitalTwinState:
    timestamp: str
    temperature: float     # Celsius (°C)
    humidity: float        # Relative Humidity (%)
    co2: float             # CO2 Concentration (ppm)
    light_intensity: float # Photosynthetically Active Radiation PAR (μmol/m²/s)
    plant_height: float    # Plant height (cm)
    step: int              # Time step index

    def to_dict(self) -> Dict[str, Any]:
        """Converts state to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Serializes state to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DigitalTwinState':
        """Constructs state object from dictionary."""
        return cls(**data)
