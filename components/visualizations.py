"""
Visualization Components for Wideband Branchline Balun Application
Creates plots and charts using Plotly and Matplotlib
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_s_parameters(sim_results: Dict, title: str = "S-Parameters") -> go.Figure:
    """
    Create S-parameter magnitude plot
    
    Args:
        sim_results: Dictionary with simulation results
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    freqs = sim_results['frequencies_ghz']
    
    # S11 - Return Loss
    fig.add_trace(go.Scatter(
        x=freqs,
        y=sim_results['s11_mag_db'],
        mode='lines',
        name='S11 (Return Loss)',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    # S21 - Insertion Loss (Port 2)
    fig.add_trace(go.Scatter(
        x=freqs,
        y=sim_results['s21_mag_db'],
        mode='lines',
        name='S21 (Output 1)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    # S31 - Insertion Loss (Port 3)
    fig.add_trace(go.Scatter(
        x=freqs,
        y=sim_results['s31_mag_db'],
        mode='lines',
        name='S31 (Output 2)',
        line=dict(color='#45B7D1', width=2)
    ))
    
    # S23 - Isolation
    fig.add_trace(go.Scatter(
        x=freqs,
        y=sim_results['s23_mag_db'],
        mode='lines',
        name='S23 (Isolation)',
        line=dict(color='#96CEB4', width=2, dash='dash')
    ))
    
    # Add reference lines
    fig.add_hline(y=-3, line_dash="dot", line_color="gray", 
                  annotation_text="-3dB")
    fig.add_hline(y=-10, line_dash="dot", line_color="orange",
                  annotation_text="-10dB (Return Loss Target)")
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        template="plotly",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    fig.update_yaxes(range=[-40, 5])
    
    return fig


def plot_phase_difference(sim_results: Dict, 
                          title: str = "Phase Difference") -> go.Figure:
    """
    Create phase difference plot between balanced outputs
    
    Args:
        sim_results: Dictionary with simulation results
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    freqs = sim_results['frequencies_ghz']
    phase_diff = np.abs(sim_results['phase_difference_deg'])
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=phase_diff,
        mode='lines',
        name='Phase Difference',
        line=dict(color='#9B59B6', width=2),
        fill='tozeroy',
        fillcolor='rgba(155, 89, 182, 0.2)'
    ))
    
    # Target line at 180°
    fig.add_hline(y=180, line_dash="dash", line_color="#2ECC71",
                  annotation_text="Target: 180°")
    
    # Tolerance band (±5°)
    fig.add_hrect(y0=175, y1=185, 
                  fillcolor="rgba(46, 204, 113, 0.1)",
                  line_width=0)
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Phase Difference (degrees)",
        template="plotly",
        height=400
    )
    
    fig.update_yaxes(range=[0, 360])
    
    return fig


def plot_vswr(sim_results: Dict, title: str = "VSWR") -> go.Figure:
    """
    Create VSWR plot
    
    Args:
        sim_results: Dictionary with simulation results
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    freqs = sim_results['frequencies_ghz']
    vswr = sim_results['vswr']
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=vswr,
        mode='lines',
        name='VSWR',
        line=dict(color='#E74C3C', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    
    # VSWR = 2 reference (acceptable limit)
    fig.add_hline(y=2, line_dash="dash", line_color="#F39C12",
                  annotation_text="VSWR = 2")
    
    # VSWR = 1.5 reference (good)
    fig.add_hline(y=1.5, line_dash="dot", line_color="#27AE60",
                  annotation_text="VSWR = 1.5")
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (GHz)",
        yaxis_title="VSWR",
        template="plotly",
        height=400
    )
    
    fig.update_yaxes(range=[1, 5])
    
    return fig


def plot_layout(dimensions: Dict, num_stages: int = 1, 
                title: str = "Balun Layout") -> go.Figure:
    """
    Create 2D layout visualization of the branchline balun
    
    Args:
        dimensions: Dictionary with calculated dimensions
        num_stages: Number of stages
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Extract dimensions
    series_w = dimensions.get('series_arm', {}).get('width_mm', 5)
    series_l = dimensions.get('series_arm', {}).get('quarter_wave_mm', 15)
    shunt_w = dimensions.get('shunt_arm', {}).get('width_mm', 3)
    shunt_l = dimensions.get('shunt_arm', {}).get('quarter_wave_mm', 15)
    
    # Scale for visibility
    scale = 1.0
    
    # Colors
    series_color = '#4ECDC4'
    shunt_color = '#FF6B6B'
    port_color = '#45B7D1'
    
    # Draw branchline coupler (simplified rectangular representation)
    # Horizontal series arms (top and bottom)
    for stage in range(num_stages):
        x_offset = stage * (series_l * 1.2)
        
        # Top horizontal arm
        fig.add_shape(
            type="rect",
            x0=x_offset, y0=shunt_l + shunt_w/2,
            x1=x_offset + series_l, y1=shunt_l + shunt_w/2 + series_w,
            fillcolor=series_color,
            line=dict(color=series_color, width=1),
        )
        
        # Bottom horizontal arm
        fig.add_shape(
            type="rect",
            x0=x_offset, y0=-series_w/2,
            x1=x_offset + series_l, y1=series_w/2,
            fillcolor=series_color,
            line=dict(color=series_color, width=1),
        )
        
        # Left vertical arm (shunt)
        fig.add_shape(
            type="rect",
            x0=x_offset - shunt_w/2, y0=0,
            x1=x_offset + shunt_w/2, y1=shunt_l + shunt_w/2,
            fillcolor=shunt_color,
            line=dict(color=shunt_color, width=1),
        )
        
        # Right vertical arm (shunt)
        fig.add_shape(
            type="rect",
            x0=x_offset + series_l - shunt_w/2, y0=0,
            x1=x_offset + series_l + shunt_w/2, y1=shunt_l + shunt_w/2,
            fillcolor=shunt_color,
            line=dict(color=shunt_color, width=1),
        )
    
    # Add port labels
    fig.add_annotation(x=-3, y=shunt_l/2, text="Port 1<br>(Input)",
                      showarrow=True, arrowhead=2, arrowcolor=port_color)
    
    total_width = num_stages * series_l * 1.2
    fig.add_annotation(x=total_width + 3, y=shunt_l + shunt_w/2, 
                      text="Port 2<br>(Output 1)",
                      showarrow=True, arrowhead=2, arrowcolor=port_color)
    fig.add_annotation(x=total_width + 3, y=0, 
                      text="Port 3<br>(Output 2)",
                      showarrow=True, arrowhead=2, arrowcolor=port_color)
    
    # Add dimension annotations
    fig.add_annotation(
        x=series_l/2, y=-5,
        text=f"λ/4 = {series_l:.2f} mm",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly",
        showlegend=False,
        height=500,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="X (mm)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Y (mm)"
        )
    )
    
    return fig


def plot_comparison(single_stage: Dict, multistage: Dict,
                    metric: str = 's11_mag_db',
                    title: str = "Comparison") -> go.Figure:
    """
    Create comparison plot between single-stage and multistage designs
    
    Args:
        single_stage: Single-stage simulation results
        multistage: Multistage simulation results
        metric: Which metric to compare
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    freqs = single_stage['frequencies_ghz']
    
    # Single stage
    fig.add_trace(go.Scatter(
        x=freqs,
        y=single_stage[metric],
        mode='lines',
        name='Conventional (Single Stage)',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # Multistage
    fig.add_trace(go.Scatter(
        x=freqs,
        y=multistage[metric],
        mode='lines',
        name='Wideband (Multistage)',
        line=dict(color='#2ECC71', width=2)
    ))
    
    # Metric-specific formatting
    metric_labels = {
        's11_mag_db': 'Return Loss (dB)',
        's21_mag_db': 'Insertion Loss S21 (dB)',
        's31_mag_db': 'Insertion Loss S31 (dB)',
        'vswr': 'VSWR',
        'phase_difference_deg': 'Phase Difference (°)'
    }
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (GHz)",
        yaxis_title=metric_labels.get(metric, metric),
        template="plotly",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=450
    )
    
    return fig


def plot_smith_chart(s11_data: List[complex], 
                     title: str = "Smith Chart") -> go.Figure:
    """
    Create a simplified Smith chart representation
    
    Args:
        s11_data: List of S11 complex values
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Draw unit circle (Smith chart boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        name='Unit Circle',
        line=dict(color='gray', width=1),
        showlegend=False
    ))
    
    # Add constant resistance circles (simplified)
    for r in [0.2, 0.5, 1, 2]:
        center = r / (1 + r)
        radius = 1 / (1 + r)
        circle_theta = np.linspace(0, 2*np.pi, 50)
        fig.add_trace(go.Scatter(
            x=center + radius * np.cos(circle_theta),
            y=radius * np.sin(circle_theta),
            mode='lines',
            line=dict(color='rgba(100,100,100,0.3)', width=0.5),
            showlegend=False
        ))
    
    # Plot S11 data points
    if s11_data:
        real_parts = [s.real for s in s11_data]
        imag_parts = [s.imag for s in s11_data]
        
        fig.add_trace(go.Scatter(
            x=real_parts,
            y=imag_parts,
            mode='lines+markers',
            name='S11',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4)
        ))
    
    # Center point (perfect match)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        name='Perfect Match',
        marker=dict(color='#2ECC71', size=10, symbol='cross')
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.2, 1.2]),
        yaxis=dict(range=[-1.2, 1.2]),
        height=500,
        width=500
    )
    
    return fig


def create_metrics_table(metrics: Dict) -> str:
    """
    Create HTML table for performance metrics
    
    Args:
        metrics: Dictionary with performance metrics
    
    Returns:
        HTML string for table
    """
    rows = []
    
    metric_labels = {
        'bandwidth_ghz': ('Bandwidth', 'GHz'),
        'bandwidth_percent': ('Bandwidth', '%'),
        'min_return_loss_db': ('Min Return Loss', 'dB'),
        'max_insertion_loss_db': ('Max Insertion Loss', 'dB'),
        'max_phase_imbalance_deg': ('Max Phase Imbalance', '°'),
        'max_amplitude_imbalance_db': ('Max Amplitude Imbalance', 'dB'),
        'min_isolation_db': ('Min Isolation', 'dB'),
        'max_vswr': ('Max VSWR', ''),
        'min_vswr': ('Min VSWR', '')
    }
    
    for key, value in metrics.items():
        if key in metric_labels:
            label, unit = metric_labels[key]
            rows.append(f"<tr><td>{label}</td><td>{value} {unit}</td></tr>")
    
    return f"""
    <table style="width:100%; border-collapse:collapse;">
        <thead>
            <tr style="background-color:#333;">
                <th style="padding:10px; text-align:left;">Metric</th>
                <th style="padding:10px; text-align:left;">Value</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

def plot_high_fidelity_layout(layout_shapes: List[Dict], title: str = "High-Fidelity CAD Layout") -> go.Figure:
    """
    Create high-fidelity layout visualization from generated polygons
    
    Args:
        layout_shapes: List of shape dictionaries from LayoutGenerator
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for shape in layout_shapes:
        fig.add_trace(go.Scatter(
            x=shape['x'],
            y=shape['y'],
            fill=shape['fill'],
            fillcolor=shape['fillcolor'],
            line=shape['line'],
            mode=shape['mode'],
            name=shape['name'],
            hoverinfo='name+x+y'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly",
        showlegend=True,
        height=600,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            title="Length (mm)",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            title="Width (mm)",
            zeroline=False
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig
