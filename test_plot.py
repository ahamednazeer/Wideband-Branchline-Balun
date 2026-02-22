import plotly.graph_objects as go
import sys
import json
sys.path.append('.')
from components.visualizations import plot_layout
import traceback

try:
    dims = {
        'series_arm': {'width_mm': 5, 'quarter_wave_mm': 15},
        'shunt_arm': {'width_mm': 3, 'quarter_wave_mm': 15}
    }
    fig = plot_layout(dims, 2)
    # Print the resulting layout min/max we can see
    print(fig.to_json())
except Exception as e:
    traceback.print_exc()
