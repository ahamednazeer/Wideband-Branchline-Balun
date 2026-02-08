"""
Layout Generator for Wideband Branchline Balun
Generates GDSII and DXF files for industry-standard fabrication and simulation
"""
import numpy as np
import gdspy
import ezdxf
import io
from typing import Dict, List, Tuple

class LayoutGenerator:
    def __init__(self, substrate_params: Dict):
        self.epsilon_r = substrate_params.get('epsilon_r', 4.4)
        self.height = substrate_params.get('thickness_mm', 1.6)
        
    def generate_layout(self, dimensions: Dict, num_stages: int = 2) -> Dict:
        """
        Generate geometry data for visualization and export
        
        Args:
            dimensions: Dictionary with calculated microstrip dimensions
            num_stages: Number of stages
            
        Returns:
            Dictionary containing polygon coordinates and export objects
        """
        # Extract dimensions (in mm)
        # We work in um for GDSII (industry standard units)
        scale = 1000.0
        
        series_w = dimensions['series_arm']['width_mm'] * scale
        series_l = dimensions['series_arm']['quarter_wave_mm'] * scale
        shunt_w = dimensions['shunt_arm']['width_mm'] * scale
        shunt_l = dimensions['shunt_arm']['quarter_wave_mm'] * scale
        
        # 50 Ohm Feed line width (approximate, for ports)
        # Using shunt width as it is usually 50 Ohm
        feed_w = shunt_w
        feed_l = series_l / 2  # Length of feed lines
        
        # Geometry containers
        polygons = []
        labels = []
        
        # Build Multistage Branchline Coupler
        # Origin (0,0) is at bottom-left feed of first stage
        
        total_length = num_stages * series_l
        
        # 1. Horizontal Arms (Series) - Color: Soft Red
        # Top Arm
        top_y = shunt_l
        polygons.append({
            'points': [
                (0, top_y - series_w/2),
                (total_length, top_y - series_w/2),
                (total_length, top_y + series_w/2),
                (0, top_y + series_w/2)
            ],
            'layer': 1,
            'desc': 'Top Series Arm',
            'color': '#ef4444' # Red-500
        })
        
        # Bottom Arm
        polygons.append({
            'points': [
                (0, -series_w/2),
                (total_length, -series_w/2),
                (total_length, series_w/2),
                (0, series_w/2)
            ],
            'layer': 1,
            'desc': 'Bottom Series Arm',
            'color': '#ef4444' # Red-500
        })
        
        # 2. Vertical Arms (Shunt) - Color: Emerald Green
        for i in range(num_stages + 1):
            x_pos = i * series_l
            polygons.append({
                'points': [
                    (x_pos - shunt_w/2, 0),
                    (x_pos + shunt_w/2, 0),
                    (x_pos + shunt_w/2, shunt_l),
                    (x_pos - shunt_w/2, shunt_l)
                ],
                'layer': 1,
                'desc': f'Shunt Arm {i+1}',
                'color': '#10b981' # Emerald-500
            })
            
        # 3. Feed Lines - Color: Blue
        # Port 1 (Input) - Bottom Left
        polygons.append({
            'points': [
                (-feed_l, -feed_w/2),
                (0, -feed_w/2),
                (0, feed_w/2),
                (-feed_l, feed_w/2)
            ],
            'layer': 1,
            'desc': 'Port 1 Feed',
            'color': '#3b82f6' # Blue-500
        })
        labels.append({'text': 'Port 1', 'pos': (-feed_l, 0)})
        
        # Port 4 (Isolated) - Top Left
        polygons.append({
            'points': [
                (-feed_l, shunt_l - feed_w/2),
                (0, shunt_l - feed_w/2),
                (0, shunt_l + feed_w/2),
                (-feed_l, shunt_l + feed_w/2)
            ],
            'layer': 1,
            'desc': 'Port 4 Feed (Term)',
            'color': '#3b82f6'
        })
        labels.append({'text': 'Port 4 (Iso)', 'pos': (-feed_l, shunt_l)})
        
        # Port 3 (Output 1 0deg) - Bottom Right
        polygons.append({
            'points': [
                (total_length, -feed_w/2),
                (total_length + feed_l, -feed_w/2),
                (total_length + feed_l, feed_w/2),
                (total_length, feed_w/2)
            ],
            'layer': 1,
            'desc': 'Port 3 Feed',
            'color': '#3b82f6'
        })
        labels.append({'text': 'Port 3 (0deg)', 'pos': (total_length + feed_l, 0)})
        
        # Port 2 (Output 2 90deg/180deg) - Top Right
        polygons.append({
            'points': [
                (total_length, shunt_l - feed_w/2),
                (total_length + feed_l, shunt_l - feed_w/2),
                (total_length + feed_l, shunt_l + feed_w/2),
                (total_length, shunt_l + feed_w/2)
            ],
            'layer': 1,
            'desc': 'Port 2 (Balanced)',
            'color': '#3b82f6'
        })
        labels.append({'text': 'Port 2 (180deg)', 'pos': (total_length + feed_l, shunt_l)})
        
        return {
            'polygons': polygons,
            'labels': labels,
            'bounds': {
                'min_x': -feed_l,
                'max_x': total_length + feed_l,
                'min_y': -series_w,
                'max_y': shunt_l + series_w
            }
        }
        
    def export_gds(self, layout_data: Dict, filename: str) -> str:
        """
        Export layout to GDSII format
        """
    def export_gds(self, layout_data: Dict, filename: str) -> str:
        """
        Export layout to GDSII format
        """
        # CRITICAL FIX: Reset the global state completely
        # This clears all previously defined cells from memory
        gdspy.current_library = gdspy.GdsLibrary()
        
        # Now we can safely use the current library
        lib = gdspy.current_library
        cell = lib.new_cell('BALUN')
        
        for poly in layout_data['polygons']:
            points = poly['points']
            gdspy_poly = gdspy.Polygon(points, layer=poly['layer'])
            cell.add(gdspy_poly)
            
        for label in layout_data['labels']:
            l = gdspy.Label(label['text'], label['pos'], layer=10)
            cell.add(l)
            
        lib.write_gds(filename)
        return filename
        
    def export_dxf(self, layout_data: Dict, filename: str) -> str:
        """
        Export layout to DXF format
        """
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        for poly in layout_data['polygons']:
            # Convert to mm for DXF (usually standard)
            # Input data is in um
            points_mm = [(p[0]/1000.0, p[1]/1000.0) for p in poly['points']]
            msp.add_lwpolyline(points_mm, close=True)
            
        doc.saveas(filename)
        return filename
        
    def get_plotly_shapes(self, layout_data: Dict) -> List[Dict]:
        """
        Convert layout data to Plotly shape dictionaries
        """
        shapes = []
        
        for poly in layout_data['polygons']:
            # Convert um back to mm for visualization
            points = np.array(poly['points']) / 1000.0
            
            # Close the polygon
            x = list(points[:, 0]) + [points[0, 0]]
            y = list(points[:, 1]) + [points[0, 1]]
            
            shapes.append({
                'type': 'scatter',
                'x': x,
                'y': y,
                'fill': 'toself',
                'fillcolor': poly.get('color', '#667eea'),
                'line': {'color': '#4a5568', 'width': 1},
                'mode': 'lines',
                'name': poly['desc']
            })
            
        return shapes
