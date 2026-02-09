"""Observables and utilities for LGT simulations.

This module provides:
- SimulationData container for logging
- Observable computation functions (plaquette, Wilson loop, VBS)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from vmc.peps.gi.local_terms import GILocalHamiltonian
    from vmc.peps.gi.model import GIPEPS

logger = logging.getLogger(__name__)


@dataclass
class SimulationData:
    """Container for simulation data to be saved."""
    
    # Metadata
    model_type: str = ""
    lattice_size: tuple[int, int] = (0, 0)
    parameters: dict = field(default_factory=dict)
    
    # Time series data
    steps: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    energy_errors: list[float] = field(default_factory=list)
    energy_variances: list[float] = field(default_factory=list)
    
    # Electric field energy (for computing dE/dg)
    electric_energies: list[float] = field(default_factory=list)
    
    # Plaquette values per site (for vison dynamics)
    plaquette_maps: list[list[list[float]]] = field(default_factory=list)
    
    # VBS order parameters (for odd-Z2)
    vbs_dx: list[float] = field(default_factory=list)
    vbs_dy: list[float] = field(default_factory=list)
    
    # Wilson loops (list of (area, value) pairs at each logged step)
    wilson_loops: list[list[tuple[int, float]]] = field(default_factory=list)
    
    def add_step(
        self,
        step: int,
        time: float,
        energy: float,
        energy_error: float = 0.0,
        energy_variance: float = 0.0,
        electric_energy: float | None = None,
    ):
        """Add basic step data."""
        self.steps.append(step)
        self.times.append(time)
        self.energies.append(energy)
        self.energy_errors.append(energy_error)
        self.energy_variances.append(energy_variance)
        if electric_energy is not None:
            self.electric_energies.append(electric_energy)
    
    def add_plaquette_map(self, plaq_map: jnp.ndarray):
        """Add plaquette expectation values for all sites."""
        self.plaquette_maps.append(plaq_map.tolist())
    
    def add_vbs_order(self, dx: float, dy: float):
        """Add VBS order parameters."""
        self.vbs_dx.append(dx)
        self.vbs_dy.append(dy)
    
    def add_wilson_loops(self, loops: list[tuple[int, float]]):
        """Add Wilson loop values for different areas."""
        self.wilson_loops.append(loops)
    
    def save(self, path: str | Path):
        """Save data to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info(f"Saved simulation data to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "SimulationData":
        """Load data from JSON file."""
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for key, value in data.items():
            setattr(obj, key, value)
        return obj


def compute_energy_derivative(g: float, electric_energy: float) -> float:
    """Compute dE/dg = (1/g) * ⟨H_E⟩.
    
    From the paper: (∂⟨H⟩/∂g) = (1/g)⟨H_E⟩
    """
    if abs(g) < 1e-10:
        return 0.0
    return electric_energy / g


def format_step_log(
    step: int,
    energy: float,
    energy_error: float,
    energy_variance: float,
    time: float | None = None,
    electric_energy: float | None = None,
    g: float | None = None,
) -> str:
    """Format a step log message."""
    parts = [f"Step {step:4d}"]
    if time is not None:
        parts.append(f"t={time:.3f}")
    parts.append(f"E = {energy:.6f} ± {energy_error:.4f} [σ²={energy_variance:.4f}]")
    if electric_energy is not None and g is not None:
        de_dg = compute_energy_derivative(g, electric_energy)
        parts.append(f"dE/dg = {de_dg:.4f}")
    return " | ".join(parts)
