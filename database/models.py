"""
SQLAlchemy models for Wideband Branchline Balun Design Application
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DesignProject(Base):
    """Main project table storing design metadata"""
    __tablename__ = 'design_projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parameters = relationship("DesignParameters", back_populates="project", uselist=False, cascade="all, delete-orphan")
    simulations = relationship("SimulationResult", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DesignProject(id={self.id}, name='{self.name}')>"


class SubstrateMaterial(Base):
    """Substrate material properties database"""
    __tablename__ = 'substrate_materials'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    dielectric_constant = Column(Float, nullable=False)  # εr
    loss_tangent = Column(Float, nullable=False)  # tan δ
    thickness_mm = Column(Float, nullable=False)  # h in mm
    conductor_thickness_um = Column(Float, default=35.0)  # t in μm (copper)
    description = Column(Text)
    is_default = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<SubstrateMaterial(name='{self.name}', εr={self.dielectric_constant})>"


class DesignParameters(Base):
    """Stores all balun design parameters for a project"""
    __tablename__ = 'design_parameters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('design_projects.id'), nullable=False)
    
    # Requirements (Phase 1)
    center_frequency_ghz = Column(Float, default=2.4)
    bandwidth_percent = Column(Float, default=25.0)
    input_impedance = Column(Float, default=50.0)
    output_impedance = Column(Float, default=50.0)
    phase_difference_deg = Column(Float, default=180.0)
    
    # Technology Selection (Phase 2)
    structure_type = Column(String(50), default='branchline_balun')
    enhancement_technique = Column(String(50), default='multistage_cascading')
    num_stages = Column(Integer, default=2)
    
    # Substrate (Phase 4)
    substrate_id = Column(Integer, ForeignKey('substrate_materials.id'))
    substrate_name = Column(String(100), default='FR4')
    dielectric_constant = Column(Float, default=4.4)
    substrate_thickness_mm = Column(Float, default=1.6)
    loss_tangent = Column(Float, default=0.02)
    
    # Calculated Physical Dimensions (Phase 3 & 4)
    free_space_wavelength_mm = Column(Float)
    effective_dielectric_constant = Column(Float)
    guided_wavelength_mm = Column(Float)
    quarter_wave_length_mm = Column(Float)
    half_wave_length_mm = Column(Float)
    line_width_50ohm_mm = Column(Float)
    line_width_35ohm_mm = Column(Float)
    line_width_70ohm_mm = Column(Float)
    
    # Optimization Parameters (Phase 7)
    is_optimized = Column(Boolean, default=False)
    optimization_iterations = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("DesignProject", back_populates="parameters")
    substrate = relationship("SubstrateMaterial")
    
    def __repr__(self):
        return f"<DesignParameters(project_id={self.project_id}, f0={self.center_frequency_ghz}GHz)>"


class SimulationResult(Base):
    """Stores S-parameter simulation results"""
    __tablename__ = 'simulation_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('design_projects.id'), nullable=False)
    
    # Simulation Configuration
    simulation_type = Column(String(50), default='baseline')  # 'baseline', 'optimized', 'comparison'
    freq_start_ghz = Column(Float, default=1.8)
    freq_stop_ghz = Column(Float, default=3.0)
    num_points = Column(Integer, default=201)
    
    # Results stored as JSON arrays
    frequencies = Column(JSON)  # Array of frequency points
    s11_mag_db = Column(JSON)   # S11 magnitude in dB
    s11_phase_deg = Column(JSON)  # S11 phase in degrees
    s21_mag_db = Column(JSON)   # S21 magnitude in dB
    s21_phase_deg = Column(JSON)
    s31_mag_db = Column(JSON)   # S31 magnitude in dB
    s31_phase_deg = Column(JSON)
    s23_mag_db = Column(JSON)   # S23 (isolation) magnitude in dB
    phase_difference = Column(JSON)  # Phase difference between ports 2 and 3
    vswr = Column(JSON)         # VSWR array
    
    # Performance Metrics
    bandwidth_achieved_mhz = Column(Float)
    min_return_loss_db = Column(Float)
    max_insertion_loss_db = Column(Float)
    phase_balance_deg = Column(Float)
    isolation_db = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    project = relationship("DesignProject", back_populates="simulations")
    
    def __repr__(self):
        return f"<SimulationResult(id={self.id}, type='{self.simulation_type}')>"


def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def get_engine(db_path='balun_design.db'):
    """Create and return database engine"""
    return create_engine(f'sqlite:///{db_path}', echo=False)
