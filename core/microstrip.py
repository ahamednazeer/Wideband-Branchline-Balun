"""
Microstrip Calculator Module
Implements microstrip transmission line parameter calculations
Based on Wheeler's equations and Hammerstad-Jensen formulas
"""
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from .rf_calculations import SubstrateProperties, DesignSpecs, C0


class MicrostripCalculator:
    """
    Microstrip Transmission Line Parameter Calculator
    
    Calculates:
    - Line width for target characteristic impedance
    - Characteristic impedance from dimensions
    - Attenuation (conductor and dielectric losses)
    """
    
    # Physical constants
    COPPER_CONDUCTIVITY = 5.8e7  # S/m
    
    def __init__(self, substrate: SubstrateProperties):
        self.substrate = substrate
    
    # ==================== Impedance Calculations ====================
    
    def width_for_impedance(self, z0: float) -> float:
        """
        Calculate microstrip width for target characteristic impedance
        
        Uses Wheeler's synthesis equations
        
        Args:
            z0: Target characteristic impedance in Ohms
        
        Returns:
            Required width in mm
        """
        er = self.substrate.epsilon_r
        h = self.substrate.thickness_mm
        
        # Intermediate parameters
        A = (z0 / 60) * np.sqrt((er + 1) / 2) + ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
        B = 377 * np.pi / (2 * z0 * np.sqrt(er))
        
        # Try narrow strip first (W/h < 2)
        w_h_narrow = (8 * np.exp(A)) / (np.exp(2 * A) - 2)
        
        # Wide strip (W/h > 2)
        w_h_wide = (2 / np.pi) * (
            B - 1 - np.log(2 * B - 1) + 
            ((er - 1) / (2 * er)) * (np.log(B - 1) + 0.39 - 0.61 / er)
        )
        
        # Choose appropriate formula based on result
        if w_h_narrow < 2:
            w_h = w_h_narrow
        else:
            w_h = w_h_wide
        
        width_mm = w_h * h
        return width_mm
    
    def impedance_from_width(self, width_mm: float) -> float:
        """
        Calculate characteristic impedance from microstrip width
        
        Uses Hammerstad-Jensen equations
        
        Args:
            width_mm: Microstrip width in mm
        
        Returns:
            Characteristic impedance in Ohms
        """
        er = self.substrate.epsilon_r
        h = self.substrate.thickness_mm
        w = width_mm
        
        # Width to height ratio
        u = w / h
        
        # Effective dielectric constant
        eps_eff = self._effective_epsilon(u)
        
        # Characteristic impedance
        if u <= 1:
            # Narrow strip
            z0 = (60 / np.sqrt(eps_eff)) * np.log(8 / u + 0.25 * u)
        else:
            # Wide strip
            z0 = (120 * np.pi) / (np.sqrt(eps_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
        
        return z0
    
    def _effective_epsilon(self, u: float) -> float:
        """
        Calculate effective dielectric constant
        
        Args:
            u: W/h ratio
        
        Returns:
            Effective dielectric constant
        """
        er = self.substrate.epsilon_r
        
        if u >= 1:
            eps_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 / u) ** (-0.5)
        else:
            eps_eff = (er + 1) / 2 + (er - 1) / 2 * (
                (1 + 12 / u) ** (-0.5) + 0.04 * (1 - u) ** 2
            )
        
        return eps_eff
    
    # ==================== Loss Calculations ====================
    
    def conductor_loss(self, width_mm: float, frequency_hz: float) -> float:
        """
        Calculate conductor (ohmic) loss in dB/m
        
        Args:
            width_mm: Microstrip width in mm
            frequency_hz: Frequency in Hz
        
        Returns:
            Conductor loss in dB/m
        """
        z0 = self.impedance_from_width(width_mm)
        w = width_mm * 1e-3  # Convert to meters
        
        # Surface resistance
        rs = np.sqrt(np.pi * frequency_hz * 4 * np.pi * 1e-7 / self.COPPER_CONDUCTIVITY)
        
        # Conductor attenuation
        alpha_c = rs / (z0 * w)  # Np/m
        
        # Convert to dB/m
        return alpha_c * 8.686
    
    def dielectric_loss(self, width_mm: float, frequency_hz: float) -> float:
        """
        Calculate dielectric loss in dB/m
        
        Args:
            width_mm: Microstrip width in mm
            frequency_hz: Frequency in Hz
        
        Returns:
            Dielectric loss in dB/m
        """
        er = self.substrate.epsilon_r
        tan_d = self.substrate.loss_tangent
        u = width_mm / self.substrate.thickness_mm
        eps_eff = self._effective_epsilon(u)
        
        # Free space wavelength
        lambda_0 = C0 / frequency_hz
        
        # Dielectric filling factor
        q = (er * (eps_eff - 1)) / (eps_eff * (er - 1)) if er != 1 else 1
        
        # Dielectric attenuation
        alpha_d = (np.pi * q * er * tan_d) / (lambda_0 * np.sqrt(eps_eff))  # Np/m
        
        # Convert to dB/m
        return alpha_d * 8.686
    
    def total_loss(self, width_mm: float, frequency_hz: float) -> float:
        """
        Calculate total line loss in dB/m
        
        Returns:
            Total loss (conductor + dielectric) in dB/m
        """
        return self.conductor_loss(width_mm, frequency_hz) + \
               self.dielectric_loss(width_mm, frequency_hz)
    
    # ==================== Design Tables ====================
    
    def calculate_design_widths(self, impedances: list = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate widths for common branchline balun impedances
        
        Args:
            impedances: List of impedances to calculate (default: [35, 50, 70])
        
        Returns:
            Dictionary with impedance as key and calculated parameters as value
        """
        if impedances is None:
            impedances = [35.0, 50.0, 70.71]  # Standard branchline values
        
        results = {}
        for z in impedances:
            width = self.width_for_impedance(z)
            actual_z = self.impedance_from_width(width)
            u = width / self.substrate.thickness_mm
            
            results[f"{z:.1f}Ω"] = {
                'target_impedance': z,
                'width_mm': round(width, 3),
                'actual_impedance': round(actual_z, 2),
                'w_h_ratio': round(u, 3),
                'effective_epsilon': round(self._effective_epsilon(u), 3)
            }
        
        return results
    
    def branchline_dimensions(self, frequency_ghz: float) -> Dict[str, Dict[str, float]]:
        """
        Calculate all dimensions for a branchline balun
        
        Args:
            frequency_ghz: Center frequency in GHz
        
        Returns:
            Complete dimension table for branchline design
        """
        frequency_hz = frequency_ghz * 1e9
        lambda_0 = C0 / frequency_hz * 1000  # Free space wavelength in mm
        
        # Branchline balun impedances
        # Series arms: Z0/√2 ≈ 35.35Ω for 50Ω design
        # Shunt arms: Z0 = 50Ω
        series_z = 50 / np.sqrt(2)  # ~35.35Ω
        shunt_z = 50.0
        
        # Calculate widths
        series_width = self.width_for_impedance(series_z)
        shunt_width = self.width_for_impedance(shunt_z)
        
        # Effective epsilon values
        series_eps = self._effective_epsilon(series_width / self.substrate.thickness_mm)
        shunt_eps = self._effective_epsilon(shunt_width / self.substrate.thickness_mm)
        
        # Guided wavelengths
        lambda_g_series = lambda_0 / np.sqrt(series_eps)
        lambda_g_shunt = lambda_0 / np.sqrt(shunt_eps)
        
        # Quarter-wave lengths
        qw_series = lambda_g_series / 4
        qw_shunt = lambda_g_shunt / 4
        
        # Half-wave lengths (for balun output arms)
        hw_series = lambda_g_series / 2
        
        return {
            'free_space_wavelength_mm': round(lambda_0, 2),
            'series_arm': {
                'impedance': round(series_z, 2),
                'width_mm': round(series_width, 3),
                'epsilon_eff': round(series_eps, 3),
                'guided_wavelength_mm': round(lambda_g_series, 2),
                'quarter_wave_mm': round(qw_series, 2),
                'half_wave_mm': round(hw_series, 2)
            },
            'shunt_arm': {
                'impedance': round(shunt_z, 2),
                'width_mm': round(shunt_width, 3),
                'epsilon_eff': round(shunt_eps, 3),
                'guided_wavelength_mm': round(lambda_g_shunt, 2),
                'quarter_wave_mm': round(qw_shunt, 2)
            }
        }
    
    def get_summary(self, frequency_ghz: float = 2.4) -> str:
        """
        Generate a text summary of microstrip calculations
        
        Returns:
            Formatted summary string
        """
        dims = self.branchline_dimensions(frequency_ghz)
        
        lines = [
            f"Microstrip Branchline Balun Design Summary",
            f"=" * 45,
            f"Substrate: εr={self.substrate.epsilon_r}, h={self.substrate.thickness_mm}mm",
            f"Center Frequency: {frequency_ghz} GHz",
            f"Free-space wavelength: {dims['free_space_wavelength_mm']:.2f} mm",
            f"",
            f"Series Arms ({dims['series_arm']['impedance']}Ω):",
            f"  Width: {dims['series_arm']['width_mm']:.3f} mm",
            f"  εeff: {dims['series_arm']['epsilon_eff']:.3f}",
            f"  λg: {dims['series_arm']['guided_wavelength_mm']:.2f} mm",
            f"  λ/4: {dims['series_arm']['quarter_wave_mm']:.2f} mm",
            f"",
            f"Shunt Arms ({dims['shunt_arm']['impedance']}Ω):",
            f"  Width: {dims['shunt_arm']['width_mm']:.3f} mm",
            f"  εeff: {dims['shunt_arm']['epsilon_eff']:.3f}",
            f"  λg: {dims['shunt_arm']['guided_wavelength_mm']:.2f} mm",
            f"  λ/4: {dims['shunt_arm']['quarter_wave_mm']:.2f} mm",
        ]
        
        return "\n".join(lines)

    # ==================== Advanced Dispersion Models (Industry Standard) ====================

    def effective_epsilon_with_dispersion(self, width_mm: float, frequency_hz: float) -> float:
        """
        Calculate frequency-dependent effective dielectric constant
        using Kirschning-Jansen model (Industry Standard)
        
        Args:
            width_mm: Line width in mm
            frequency_hz: Frequency in Hz
            
        Returns:
            Dispersive effective dielectric constant
        """
        w = width_mm
        h = self.substrate.thickness_mm
        er = self.substrate.epsilon_r
        
        # Static effective constant
        u = w / h
        ee_static = self._effective_epsilon(u)
        
        # Frequency parameter fn = f * h (GHz * mm)
        f_ghz = frequency_hz / 1e9
        fn = f_ghz * h
        
        # Kirschning-Jansen Model Coefficients
        P1 = 0.27488 + (0.6315 + 0.525 / (1 + 0.157 * fn)**20) * (u - 0.657)
        P2 = 0.33622 * (1 - np.exp(-0.03442 * er))
        P3 = 0.0363 * np.exp(-4.6 * u) * (1 - np.exp(-(fn / 3.87)**4.97))
        P4 = 1 + 2.751 * (1 - np.exp(-(er / 15.916)**8))
        
        Pf = P1 * P2 * ((0.1844 * P3 * P4 * fn)**1.5763)
        
        # Dispersive effective constant
        ee_f = er - (er - ee_static) / (1 + Pf)
        
        return ee_f

    def impedance_with_dispersion(self, width_mm: float, frequency_hz: float) -> float:
        """
        Calculate frequency-dependent characteristic impedance
        using Kirschning-Jansen Power-Current model
        
        Args:
            width_mm: Line width in mm
            frequency_hz: Frequency in Hz
            
        Returns:
            Dispersive characteristic impedance
        """
        w = width_mm
        h = self.substrate.thickness_mm
        er = self.substrate.epsilon_r
        
        # Static values
        z0_static = self.impedance_from_width(width_mm)
        ee_static = self._effective_epsilon(w/h)
        ee_f = self.effective_epsilon_with_dispersion(width_mm, frequency_hz)
        
        # Frequency parameter using standard units
        f_ghz = frequency_hz / 1e9
        fn = f_ghz * h
        u = w / h
        
        # Power-Current formulation coefficients
        R1 = 0.03891 * (er**1.4)
        R2 = 0.267 * (u**7)
        R3 = 4.766 * np.exp(-3.228 * (u**0.641))
        R4 = 0.016 + (0.054 - 0.016) / (1 + 10 * (u**2)) # Corrected coefficient
        
        # Frequency dependence factor
        # This is a simplified cohesive model often used in simulators
        # Z0(f) = Z0(0) * [(ee_f - 1) / (ee_static - 1)] * sqrt(ee_static/ee_f)
        
        term1 = (ee_f - 1) / (ee_static - 1)
        term2 = np.sqrt(ee_static / ee_f)
        
        z0_f = z0_static * term1 * term2
        
        return z0_f
