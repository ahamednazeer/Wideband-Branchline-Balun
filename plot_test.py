import plotly.graph_objects as go
import sys
sys.path.append('.')
from components.visualizations import plot_layout
from core.rf_calculations import SubstrateProperties
from core.microstrip import MicrostripCalculator

substrate = SubstrateProperties(epsilon_r=4.4, thickness_mm=1.6, loss_tangent=0.02)
microstrip = MicrostripCalculator(substrate)
dims = microstrip.branchline_dimensions(2.4)
fig = plot_layout(dims, 2)
fig.write_image("test_layout.png")
