"""
Database Manager for Wideband Branchline Balun Application
Handles all database operations with SQLite
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List
from .models import Base, DesignProject, DesignParameters, SubstrateMaterial, SimulationResult


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, 'data', 'balun_design.db')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Initialize default substrates
        self._init_default_substrates()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def _init_default_substrates(self):
        """Initialize default substrate materials"""
        session = self.get_session()
        try:
            # Check if substrates already exist
            existing = session.query(SubstrateMaterial).first()
            if existing:
                return
            
            # Default substrate materials
            substrates = [
                SubstrateMaterial(
                    name="FR4",
                    dielectric_constant=4.4,
                    loss_tangent=0.02,
                    thickness_mm=1.6,
                    conductor_thickness_um=35.0,
                    description="Standard FR4 PCB material - cost-effective for academic use",
                    is_default=True
                ),
                SubstrateMaterial(
                    name="Rogers RO4003C",
                    dielectric_constant=3.55,
                    loss_tangent=0.0027,
                    thickness_mm=0.813,
                    conductor_thickness_um=35.0,
                    description="Low-loss thermoset laminate for RF applications"
                ),
                SubstrateMaterial(
                    name="Rogers RO4350B",
                    dielectric_constant=3.48,
                    loss_tangent=0.0037,
                    thickness_mm=0.762,
                    conductor_thickness_um=35.0,
                    description="High-frequency laminate with glass-reinforced hydrocarbon"
                ),
                SubstrateMaterial(
                    name="Rogers RT/duroid 5880",
                    dielectric_constant=2.2,
                    loss_tangent=0.0009,
                    thickness_mm=0.787,
                    conductor_thickness_um=35.0,
                    description="PTFE composite for high-frequency applications"
                ),
                SubstrateMaterial(
                    name="Taconic TLY-5",
                    dielectric_constant=2.2,
                    loss_tangent=0.0009,
                    thickness_mm=0.508,
                    conductor_thickness_um=35.0,
                    description="Woven fiberglass reinforced PTFE"
                ),
            ]
            
            for substrate in substrates:
                session.add(substrate)
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error initializing substrates: {e}")
        finally:
            session.close()
    
    # ==================== Project Operations ====================
    
    def create_project(self, name: str, description: str = "") -> DesignProject:
        """Create a new design project"""
        session = self.get_session()
        try:
            project = DesignProject(name=name, description=description)
            session.add(project)
            session.commit()
            session.refresh(project)
            return project
        finally:
            session.close()
    
    def get_project(self, project_id: int) -> Optional[DesignProject]:
        """Get a project by ID"""
        session = self.get_session()
        try:
            return session.query(DesignProject).filter_by(id=project_id).first()
        finally:
            session.close()
    
    def get_all_projects(self) -> List[DesignProject]:
        """Get all projects"""
        session = self.get_session()
        try:
            return session.query(DesignProject).order_by(DesignProject.updated_at.desc()).all()
        finally:
            session.close()
    
    def update_project(self, project_id: int, **kwargs) -> Optional[DesignProject]:
        """Update project fields"""
        session = self.get_session()
        try:
            project = session.query(DesignProject).filter_by(id=project_id).first()
            if project:
                for key, value in kwargs.items():
                    if hasattr(project, key):
                        setattr(project, key, value)
                session.commit()
                session.refresh(project)
            return project
        finally:
            session.close()
    
    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all related data"""
        session = self.get_session()
        try:
            project = session.query(DesignProject).filter_by(id=project_id).first()
            if project:
                session.delete(project)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # ==================== Design Parameters Operations ====================
    
    def save_parameters(self, project_id: int, **kwargs) -> DesignParameters:
        """Save or update design parameters for a project"""
        session = self.get_session()
        try:
            params = session.query(DesignParameters).filter_by(project_id=project_id).first()
            
            if params is None:
                params = DesignParameters(project_id=project_id)
                session.add(params)
            
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            session.commit()
            session.refresh(params)
            return params
        finally:
            session.close()
    
    def get_parameters(self, project_id: int) -> Optional[DesignParameters]:
        """Get design parameters for a project"""
        session = self.get_session()
        try:
            return session.query(DesignParameters).filter_by(project_id=project_id).first()
        finally:
            session.close()
    
    # ==================== Substrate Operations ====================
    
    def get_all_substrates(self) -> List[SubstrateMaterial]:
        """Get all substrate materials"""
        session = self.get_session()
        try:
            return session.query(SubstrateMaterial).all()
        finally:
            session.close()
    
    def get_substrate_by_name(self, name: str) -> Optional[SubstrateMaterial]:
        """Get substrate by name"""
        session = self.get_session()
        try:
            return session.query(SubstrateMaterial).filter_by(name=name).first()
        finally:
            session.close()
    
    def get_default_substrate(self) -> Optional[SubstrateMaterial]:
        """Get the default substrate (FR4)"""
        session = self.get_session()
        try:
            return session.query(SubstrateMaterial).filter_by(is_default=True).first()
        finally:
            session.close()
    
    # ==================== Simulation Results Operations ====================
    
    def save_simulation(self, project_id: int, simulation_type: str, **kwargs) -> SimulationResult:
        """Save simulation results"""
        session = self.get_session()
        try:
            sim = SimulationResult(
                project_id=project_id,
                simulation_type=simulation_type,
                **kwargs
            )
            session.add(sim)
            session.commit()
            session.refresh(sim)
            return sim
        finally:
            session.close()
    
    def get_simulations(self, project_id: int, simulation_type: str = None) -> List[SimulationResult]:
        """Get simulation results for a project"""
        session = self.get_session()
        try:
            query = session.query(SimulationResult).filter_by(project_id=project_id)
            if simulation_type:
                query = query.filter_by(simulation_type=simulation_type)
            return query.order_by(SimulationResult.created_at.desc()).all()
        finally:
            session.close()
    
    def get_latest_simulation(self, project_id: int, simulation_type: str = None) -> Optional[SimulationResult]:
        """Get the most recent simulation for a project"""
        session = self.get_session()
        try:
            query = session.query(SimulationResult).filter_by(project_id=project_id)
            if simulation_type:
                query = query.filter_by(simulation_type=simulation_type)
            return query.order_by(SimulationResult.created_at.desc()).first()
        finally:
            session.close()
    
    def delete_simulation(self, simulation_id: int) -> bool:
        """Delete a simulation result"""
        session = self.get_session()
        try:
            sim = session.query(SimulationResult).filter_by(id=simulation_id).first()
            if sim:
                session.delete(sim)
                session.commit()
                return True
            return False
        finally:
            session.close()
