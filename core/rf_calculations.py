"""
RF Calculations Module for Wideband Branchline Balun
Implements transmission line theory and fundamental RF equations
"""
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


# Physical Constants
C0 = 299792458  # Speed of light in m/s
MU0 = 4 * np.pi * 1e-7  # Permeability of free space
EPS0 = 8.854187817e-12  # Permittivity of free space


@dataclass
class SubstrateProperties:
    """Substrate material properties"""
    epsilon_r: float  # Relative dielectric constant
    thickness_mm: float  # Substrate height in mm
    loss_tangent: float  # tan δ
    conductor_thickness_um: float = 35.0  # Copper thickness in μm
    
    @property
    def thickness_m(self) -> float:
        return self.thickness_mm * 1e-3
    
    @property
    def conductor_thickness_m(self) -> float:
        return self.conductor_thickness_um * 1e-6


@dataclass
class DesignSpecs:
    """Design specifications"""
    center_frequency_ghz: float
    input_impedance: float = 50.0
    output_impedance: float = 50.0
    bandwidth_percent: float = 25.0
    phase_difference_deg: float = 180.0
    num_stages: int = 2
    
    @property
    def center_frequency_hz(self) -> float:
        return self.center_frequency_ghz * 1e9
    
    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Return frequency range in Hz"""
        bw_factor = self.bandwidth_percent / 100 / 2
        f_low = self.center_frequency_hz * (1 - bw_factor)
        f_high = self.center_frequency_hz * (1 + bw_factor)
        return f_low, f_high


class RFCalculator:
    """
    RF Transmission Line Calculator
    Implements core equations for microwave design
    """
    
    def __init__(self, substrate: SubstrateProperties, specs: DesignSpecs):
        self.substrate = substrate
        self.specs = specs
    
    # ==================== Wavelength Calculations ====================
    
    def free_space_wavelength(self, frequency_hz: float = None) -> float:
        """
        Calculate free-space wavelength λ₀ = c/f
        
        Args:
            frequency_hz: Frequency in Hz (default: center frequency)
        
        Returns:
            Wavelength in meters
        """
        if frequency_hz is None:
            frequency_hz = self.specs.center_frequency_hz
        return C0 / frequency_hz
    
    def free_space_wavelength_mm(self, frequency_hz: float = None) -> float:
        """Return free-space wavelength in mm"""
        return self.free_space_wavelength(frequency_hz) * 1000
    
    def effective_dielectric_constant(self, width_mm: float) -> float:
        """
        Calculate effective dielectric constant for microstrip
        
        εeff = (εr + 1)/2 + (εr - 1)/2 × (1 + 12h/W)^(-0.5)
        
        This is the Hammerstad-Jensen formula for effective εr
        
        Args:
            width_mm: Microstrip width in mm
        
        Returns:
            Effective dielectric constant
        """
        er = self.substrate.epsilon_r
        h = self.substrate.thickness_mm
        w = width_mm
        
        # Handle edge cases
        if w <= 0:
            return er
        
        ratio = w / h
        
        if ratio >= 1:
            # Wide strip formula
            eps_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / w) ** (-0.5)
        else:
            # Narrow strip formula
            eps_eff = (er + 1) / 2 + (er - 1) / 2 * (
                (1 + 12 * h / w) ** (-0.5) + 0.04 * (1 - w / h) ** 2
            )
        
        return eps_eff
    
    def guided_wavelength(self, width_mm: float, frequency_hz: float = None) -> float:
        """
        Calculate guided wavelength in microstrip
        
        λg = λ₀ / √εeff
        
        Args:
            width_mm: Microstrip width in mm
            frequency_hz: Frequency in Hz
        
        Returns:
            Guided wavelength in meters
        """
        lambda_0 = self.free_space_wavelength(frequency_hz)
        eps_eff = self.effective_dielectric_constant(width_mm)
        return lambda_0 / np.sqrt(eps_eff)
    
    def guided_wavelength_mm(self, width_mm: float, frequency_hz: float = None) -> float:
        """Return guided wavelength in mm"""
        return self.guided_wavelength(width_mm, frequency_hz) * 1000
    
    # ==================== Section Length Calculations ====================
    
    def quarter_wave_length(self, width_mm: float, frequency_hz: float = None) -> float:
        """
        Calculate quarter-wave length for microstrip
        
        L = λg / 4
        
        Returns:
            Quarter-wave length in meters
        """
        return self.guided_wavelength(width_mm, frequency_hz) / 4
    
    def quarter_wave_length_mm(self, width_mm: float, frequency_hz: float = None) -> float:
        """Return quarter-wave length in mm"""
        return self.quarter_wave_length(width_mm, frequency_hz) * 1000
    
    def half_wave_length(self, width_mm: float, frequency_hz: float = None) -> float:
        """Calculate half-wave length in meters"""
        return self.guided_wavelength(width_mm, frequency_hz) / 2
    
    def half_wave_length_mm(self, width_mm: float, frequency_hz: float = None) -> float:
        """Return half-wave length in mm"""
        return self.half_wave_length(width_mm, frequency_hz) * 1000
    
    # ==================== Phase Calculations ====================
    
    def phase_constant(self, width_mm: float, frequency_hz: float = None) -> float:
        """
        Calculate phase constant β = 2π/λg
        
        Returns:
            Phase constant in rad/m
        """
        lambda_g = self.guided_wavelength(width_mm, frequency_hz)
        return 2 * np.pi / lambda_g
    
    def electrical_length(self, physical_length_mm: float, width_mm: float, 
                         frequency_hz: float = None) -> float:
        """
        Calculate electrical length in degrees
        
        θ = β × L × (180/π)
        
        Args:
            physical_length_mm: Physical length in mm
            width_mm: Line width in mm
            frequency_hz: Frequency in Hz
        
        Returns:
            Electrical length in degrees
        """
        beta = self.phase_constant(width_mm, frequency_hz)
        length_m = physical_length_mm * 1e-3
        return beta * length_m * 180 / np.pi
    
    # ==================== Design Summary ====================
    
    def calculate_all(self, line_width_mm: float = None) -> Dict[str, Any]:
        """
        Calculate all RF parameters at center frequency
        
        Args:
            line_width_mm: Microstrip width (if None, will calculate for 50Ω)
        
        Returns:
            Dictionary with all calculated parameters
        """
        if line_width_mm is None:
            # Use approximate 50Ω width for FR4
            line_width_mm = 3.0  # Will be refined by MicrostripCalculator
        
        freq = self.specs.center_frequency_hz
        
        return {
            'center_frequency_ghz': self.specs.center_frequency_ghz,
            'free_space_wavelength_mm': self.free_space_wavelength_mm(freq),
            'effective_dielectric_constant': self.effective_dielectric_constant(line_width_mm),
            'guided_wavelength_mm': self.guided_wavelength_mm(line_width_mm, freq),
            'quarter_wave_length_mm': self.quarter_wave_length_mm(line_width_mm, freq),
            'half_wave_length_mm': self.half_wave_length_mm(line_width_mm, freq),
            'phase_constant_rad_m': self.phase_constant(line_width_mm, freq),
            'bandwidth_percent': self.specs.bandwidth_percent,
            'frequency_range_ghz': (
                self.specs.frequency_range[0] / 1e9,
                self.specs.frequency_range[1] / 1e9
            )
        }
    
    def get_design_table(self, line_widths: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Generate design table for multiple impedance sections
        
        Args:
            line_widths: Dict of impedance labels to widths (e.g., {'50Ω': 3.0, '35Ω': 5.0})
        
        Returns:
            Nested dict with calculations for each impedance
        """
        result = {}
        for label, width in line_widths.items():
            result[label] = {
                'width_mm': width,
                'effective_epsilon': self.effective_dielectric_constant(width),
                'guided_wavelength_mm': self.guided_wavelength_mm(width),
                'quarter_wave_mm': self.quarter_wave_length_mm(width),
                'half_wave_mm': self.half_wave_length_mm(width),
            }
        return result


def calculate_vswr(gamma: complex) -> float:
    """
    Calculate VSWR from reflection coefficient
    
    VSWR = (1 + |Γ|) / (1 - |Γ|)
    """
    gamma_mag = abs(gamma)
    if gamma_mag >= 1:
        return float('inf')
    return (1 + gamma_mag) / (1 - gamma_mag)


def db_to_linear(db_value: float) -> float:
    """Convert dB to linear (power)"""
    return 10 ** (db_value / 10)


def linear_to_db(linear_value: float) -> float:
    """Convert linear (power) to dB"""
    if linear_value <= 0:
        return -100  # Floor value
    return 10 * np.log10(linear_value)


def mag_to_db(magnitude: float) -> float:
    """Convert voltage magnitude to dB"""
    if magnitude <= 0:
        return -100
    return 20 * np.log10(magnitude)


def phase_wrap(phase_deg: float) -> float:
    """Wrap phase to [-180, 180] degrees"""
    return ((phase_deg + 180) % 360) - 180
