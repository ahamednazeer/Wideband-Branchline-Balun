import sys
sys.path.append('.')
from core.rf_calculations import RFCalculator, SubstrateProperties, DesignSpecs
from core.microstrip import MicrostripCalculator
from core.simulation import BalunSimulator

substrate = SubstrateProperties(epsilon_r=4.4, thickness_mm=1.6, loss_tangent=0.02)
microstrip = MicrostripCalculator(substrate)
dims = microstrip.branchline_dimensions(2.4)
print(dims)
