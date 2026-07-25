"""
Procedural L-System Generator for Plant Architecture.
Implements Lindenmayer String Expansion rules driven by Digital Twin plant growth state.
"""

from typing import Dict

class LSystemGenerator:
    """Generates procedural L-System string representations for plant morphological structure."""

    def __init__(self, axiom: str = "X", rules: Dict[str, str] = None):
        self.axiom = axiom
        if rules is None:
            # Standard stochastic/fractal plant branching rules
            self.rules = {
                "X": "F[+X][-X]FX",
                "F": "FF"
            }
        else:
            self.rules = rules

    def generate_string(self, iterations: int) -> str:
        """Expands axiom according to production rules for N iterations."""
        current = self.axiom
        for _ in range(iterations):
            next_str = []
            for char in current:
                next_str.append(self.rules.get(char, char))
            current = "".join(next_str)
        return current

    def get_iteration_depth_from_height(self, plant_height_cm: float) -> int:
        """Maps plant height (cm) to appropriate L-System expansion iteration depth [1 - 4]."""
        if plant_height_cm < 20.0:
            return 1
        elif plant_height_cm < 60.0:
            return 2
        elif plant_height_cm < 120.0:
            return 3
        else:
            return 4
