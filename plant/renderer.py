"""
Plant Geometric Renderer.
Converts L-System string representations into 2D and 3D Plotly visualisations
integrated directly with the Streamlit dashboard and Digital Twin state.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Tuple, List, Dict
from plant.lsystem import LSystemGenerator

class PlantRenderer:
    """Renders procedural L-System plant structures in 2D and 3D using Plotly."""

    def __init__(self, branching_angle: float = 25.0):
        self.branching_angle_rad = np.radians(branching_angle)
        self.lsystem = LSystemGenerator()

    def generate_turtle_geometry_2d(self, lsystem_str: str, base_step_len: float = 1.0) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float]]]:
        """
        Parses L-System string into 2D line segments [(x1, y1, x2, y2)] and leaf positions [(x, y)].
        Commands:
          'F': Move forward and draw stem segment
          '+': Turn right by branching angle
          '-': Turn left by branching angle
          '[': Push current state (position, angle) onto stack
          ']': Pop state from stack
        """
        stack = []
        x, y = 0.0, 0.0
        angle = np.pi / 2.0  # Pointing vertically upwards (90 degrees)

        segments = []
        leaves = []

        for char in lsystem_str:
            if char == "F":
                dx = base_step_len * np.cos(angle)
                dy = base_step_len * np.sin(angle)
                new_x = x + dx
                new_y = y + dy
                segments.append((x, y, new_x, new_y))
                x, y = new_x, new_y
            elif char == "+":
                angle -= self.branching_angle_rad
            elif char == "-":
                angle += self.branching_angle_rad
            elif char == "[":
                stack.append((x, y, angle))
            elif char == "]":
                leaves.append((x, y))
                if stack:
                    x, y, angle = stack.pop()

        return segments, leaves

    def render_plotly_2d(self, plant_height_cm: float) -> go.Figure:
        """Renders interactive 2D plant structure conditioned on Digital Twin plant height."""
        iterations = self.lsystem.get_iteration_depth_from_height(plant_height_cm)
        lsystem_str = self.lsystem.generate_string(iterations)

        # Scale step length proportionally to plant height
        base_step_len = max(0.5, plant_height_cm / (10.0 * (iterations ** 1.5)))
        segments, leaves = self.generate_turtle_geometry_2d(lsystem_str, base_step_len)

        fig = go.Figure()

        # Add Stem / Branch line segments
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in segments:
            x_coords.extend([x1, x2, None])
            y_coords.extend([y1, y2, None])

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='#2e7d32', width=max(2, int(4 - iterations*0.5))),
            name='Stems & Branches',
            hoverinfo='none'
        ))

        # Add Foliage/Leaf nodes at branch tips
        if leaves:
            lx, ly = zip(*leaves)
            fig.add_trace(go.Scatter(
                x=lx,
                y=ly,
                mode='markers',
                marker=dict(size=max(4, int(10 - iterations)), color='#66bb6a', symbol='circle'),
                name='Foliage Nodes'
            ))

        fig.update_layout(
            title=f"Procedural L-System Plant Growth Structure (Height: {plant_height_cm:.1f} cm | Depth: {iterations})",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )

        return fig
