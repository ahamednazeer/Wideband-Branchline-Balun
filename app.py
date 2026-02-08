"""
Wideband Branchline Balun Design & Simulation Application
Enhanced UI/UX Version
"""
import streamlit as st
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from core.rf_calculations import RFCalculator, SubstrateProperties, DesignSpecs
from core.microstrip import MicrostripCalculator
from core.simulation import BalunSimulator, SimulationConfig
from core.optimization import MultistageOptimizer
from components.visualizations import (
    plot_s_parameters, plot_phase_difference, plot_vswr,
    plot_layout, plot_comparison
)

# Page configuration
st.set_page_config(
    page_title="Wideband Branchline Balun Designer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937; /* Gray-900 */
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -0.025em;
    }
    
    /* Dark mode support for header */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #f3f4f6; /* Gray-100 */
        }
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280; /* Gray-500 */
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }
    
    /* Cards - Adaptive */
    .glass-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb; /* Gray-200 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    
    /* Dark mode support for cards */
    @media (prefers-color-scheme: dark) {
        .glass-card {
            background-color: #1f2937; /* Gray-800 */
            border-color: #374151; /* Gray-700 */
            box-shadow: none;
        }
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #f9fafb; /* Gray-50 */
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #111827; /* Gray-900 */
            border-color: #374151;
        }
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb; /* Blue-600 */
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .section-header {
            color: #f3f4f6;
        }
    }
    
    /* Data Table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    
    .data-table th {
        text-align: left;
        padding: 0.75rem 1rem;
        color: #6b7280;
        font-weight: 600;
        border-bottom: 2px solid #e5e7eb;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    .data-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e5e7eb;
        color: #374151;
        font-family: 'JetBrains Mono', monospace;
    }
    
    @media (prefers-color-scheme: dark) {
        .data-table th {
            border-bottom-color: #374151;
            color: #9ca3af;
        }
        .data-table td {
            border-bottom-color: #374151;
            color: #d1d5db;
        }
    }
    
    /* Formula Box */
    .formula-box {
        background-color: #f3f4f6;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        color: #1f2937;
        border-radius: 0 4px 4px 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .formula-box {
            background-color: #374151;
            color: #f3f4f6;
        }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# Main Header
st.markdown('<h1 class="main-header">Wideband Branchline Balun Designer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional RF Design & Simulation Tool</p>', unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### ⚙️ Design Parameters")
    
    # Frequency Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">📻 Frequency Settings</p>', unsafe_allow_html=True)
    center_freq = st.number_input(
        "Center Frequency (GHz)", 
        min_value=0.1, max_value=30.0, 
        value=2.4, step=0.1,
        help="Design center frequency for the balun"
    )
    bandwidth = st.slider(
        "Target Bandwidth (%)", 
        min_value=5, max_value=50, value=25,
        help="Minimum bandwidth requirement"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Impedance Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">🔌 Impedance</p>', unsafe_allow_html=True)
    z0 = st.number_input(
        "Characteristic Impedance (Ω)", 
        min_value=10.0, max_value=200.0, 
        value=50.0, step=1.0
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Substrate Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">📐 Substrate Material</p>', unsafe_allow_html=True)
    
    substrate_options = {
        "FR4 Standard": (4.4, 1.6, 0.02),
        "Rogers RO4003C": (3.55, 0.813, 0.0027),
        "Rogers RO4350B": (3.48, 0.762, 0.0037),
        "Rogers RT/duroid 5880": (2.2, 0.787, 0.0009),
        "Custom...": None
    }
    
    substrate_choice = st.selectbox("Material", list(substrate_options.keys()))
    
    if substrate_choice == "Custom...":
        col1, col2 = st.columns(2)
        with col1:
            epsilon_r = st.number_input("εr", min_value=1.0, max_value=15.0, value=4.4)
        with col2:
            thickness = st.number_input("h (mm)", min_value=0.1, max_value=5.0, value=1.6)
        loss_tan = st.number_input("tan δ", min_value=0.0001, max_value=0.1, value=0.02, format="%.4f")
    else:
        epsilon_r, thickness, loss_tan = substrate_options[substrate_choice]
        # Show substrate info
        st.caption(f"εr = {epsilon_r} | h = {thickness}mm | tan δ = {loss_tan}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">🔧 Configuration</p>', unsafe_allow_html=True)
    num_stages = st.radio("Number of Stages", [1, 2, 3], index=1, horizontal=True)
    
    stage_info = {
        1: ("Conventional", "badge-warning"),
        2: ("Wideband", "badge-success"),
        3: ("Ultra-Wide", "badge-info")
    }
    info_text, badge_class = stage_info[num_stages]
    st.markdown(f'<span class="{badge_class}">{info_text}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 Reset Defaults", use_container_width=True, type="secondary"):
        st.rerun()

# Create design objects
substrate = SubstrateProperties(
    epsilon_r=epsilon_r,
    thickness_mm=thickness,
    loss_tangent=loss_tan
)

specs = DesignSpecs(
    center_frequency_ghz=center_freq,
    input_impedance=z0,
    output_impedance=z0,
    bandwidth_percent=bandwidth,
    num_stages=num_stages
)

rf_calc = RFCalculator(substrate, specs)
microstrip = MicrostripCalculator(substrate)
simulator = BalunSimulator(substrate, specs)

# Quick Stats Bar
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('''
    <div class="metric-card">
        <div class="metric-value">''' + f'{center_freq}' + '''<span style="font-size:1rem;color:#9ca3af;"> GHz</span></div>
        <div class="metric-label">Center Frequency</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="metric-card">
        <div class="metric-value">≥''' + f'{bandwidth}' + '''<span style="font-size:1rem;color:#9ca3af;">%</span></div>
        <div class="metric-label">Target Bandwidth</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="metric-card">
        <div class="metric-value">''' + f'{z0:.0f}' + '''<span style="font-size:1rem;color:#9ca3af;"> Ω</span></div>
        <div class="metric-label">Impedance</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown('''
    <div class="metric-card">
        <div class="metric-value">''' + f'{num_stages}' + '''</div>
        <div class="metric-label">Stages</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📐 Design Calculator", 
    "📊 Simulation", 
    "🎲 Yield Analysis",
    "⚖️ Comparison",
    "📋 Report",
    "📚 Theory",
    "💾 CAD Export"
])

# ==================== TAB 1: DESIGN CALCULATOR ====================
with tab1:
    st.markdown('<div class="section-header">📐 Microstrip Dimensions Calculator</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        # Get branchline dimensions
        dimensions = microstrip.branchline_dimensions(center_freq)
        lambda_0 = dimensions['free_space_wavelength_mm']
        series = dimensions['series_arm']
        shunt = dimensions['shunt_arm']
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#667eea;margin:0 0 1rem 0;">🌊 Wavelength Calculations</h4>
            <table class="data-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Free-space wavelength (λ₀)</td><td><strong>''' + f'{lambda_0:.2f}' + ''' mm</strong></td></tr>
                <tr><td>Guided λg (Series)</td><td>''' + f'{series["guided_wavelength_mm"]:.2f}' + ''' mm</td></tr>
                <tr><td>Guided λg (Shunt)</td><td>''' + f'{shunt["guided_wavelength_mm"]:.2f}' + ''' mm</td></tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#10b981;margin:0 0 1rem 0;">📏 Series Arms (Z₀/√2 ≈ 35.35Ω)</h4>
            <table class="data-table">
                <tr><th>Dimension</th><th>Value</th></tr>
                <tr><td>Width</td><td><strong>''' + f'{series["width_mm"]:.3f}' + ''' mm</strong></td></tr>
                <tr><td>Length (λ/4)</td><td><strong>''' + f'{series["quarter_wave_mm"]:.2f}' + ''' mm</strong></td></tr>
                <tr><td>Effective εr</td><td>''' + f'{series["epsilon_eff"]:.3f}' + '''</td></tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#f59e0b;margin:0 0 1rem 0;">📏 Shunt Arms (50Ω)</h4>
            <table class="data-table">
                <tr><th>Dimension</th><th>Value</th></tr>
                <tr><td>Width</td><td><strong>''' + f'{shunt["width_mm"]:.3f}' + ''' mm</strong></td></tr>
                <tr><td>Length (λ/4)</td><td><strong>''' + f'{shunt["quarter_wave_mm"]:.2f}' + ''' mm</strong></td></tr>
                <tr><td>Effective εr</td><td>''' + f'{shunt["epsilon_eff"]:.3f}' + '''</td></tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"#### 🎨 Layout Preview ({num_stages}-Stage)")
        layout_fig = plot_layout(dimensions, num_stages, "")
        layout_fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,46,0.5)'
        )
        st.plotly_chart(layout_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: SIMULATION ====================
with tab2:
    st.markdown('<div class="section-header">📊 S-Parameter Simulation (Advanced)</div>', unsafe_allow_html=True)
    
    # Simulation config in a nice card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])
    with col1:
        freq_start = st.number_input("Start (GHz)", value=1.8, step=0.1, key="sim_start")
    with col2:
        freq_stop = st.number_input("Stop (GHz)", value=3.0, step=0.1, key="sim_stop")
    with col3:
        num_points = st.selectbox("Points", [101, 201, 401], index=1, key="sim_points")
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
    
    # Advanced Options with tooltips
    with st.expander("⚙️ Advanced Modeling Options"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            use_dispersion = st.checkbox("Kirschning-Jansen Dispersion", value=True, 
                                        help="Enable frequency-dependent effective dielectric constant and impedance modeling")
        with col_adv2:
            use_discontinuities = st.checkbox("Junction Discontinuities", value=True, 
                                             help="Include T-junction parasitic effects and reference plane shifts")
    st.markdown('</div>', unsafe_allow_html=True)
    
    sim_config = SimulationConfig(
        freq_start_ghz=freq_start,
        freq_stop_ghz=freq_stop,
        num_points=num_points
    )
    
    if run_sim:
        with st.spinner("🔄 Running high-fidelity simulation..."):
            import time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            if num_stages == 1:
                sim_results = simulator.simulate_single_stage_balun(sim_config)
            else:
                sim_results = simulator.simulate_multistage_balun(
                    num_stages, 
                    sim_config,
                    use_dispersion=use_dispersion,
                    use_discontinuities=use_discontinuities
                )
            
            st.session_state['sim_results'] = sim_results
            st.session_state['sim_metrics'] = simulator.calculate_performance_metrics(sim_results)
            progress_bar.empty()
        st.success("✅ Simulation complete! (Dispersion: " + ("ON" if use_dispersion else "OFF") + ")")
    
    # Display results if available
    if 'sim_results' in st.session_state:
        sim_results = st.session_state['sim_results']
        metrics = st.session_state['sim_metrics']
        
        # Performance metrics in cards
        st.markdown('<div class="section-header">📈 Performance Metrics</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        
        bw_status = "✅" if metrics['bandwidth_percent'] >= bandwidth else "⚠️"
        rl_status = "✅" if metrics['min_return_loss_db'] < -10 else "⚠️"
        phase_status = "✅" if metrics['max_phase_imbalance_deg'] < 10 else "⚠️"
        vswr_status = "✅" if metrics['max_vswr'] < 2 else "⚠️"
        
        with m1:
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#10b981;">{metrics['bandwidth_percent']:.1f}%</div>
                <div style="color:#9ca3af;font-size:0.9rem;">Bandwidth {bw_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#3b82f6;">{metrics['min_return_loss_db']:.1f} dB</div>
                <div style="color:#9ca3af;font-size:0.9rem;">Return Loss {rl_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        with m3:
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#f59e0b;">{metrics['max_phase_imbalance_deg']:.1f}°</div>
                <div style="color:#9ca3af;font-size:0.9rem;">Phase Imbalance {phase_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        with m4:
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#ec4899;">{metrics['max_vswr']:.2f}</div>
                <div style="color:#9ca3af;font-size:0.9rem;">Max VSWR {vswr_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Plots
        st.markdown('<div class="section-header">📉 S-Parameter Charts</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_s = plot_s_parameters(sim_results, "S-Parameters")
            fig_s.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,46,0.3)')
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_p = plot_phase_difference(sim_results, "Phase Difference")
            fig_p.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,46,0.3)')
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_v = plot_vswr(sim_results, "VSWR")
        fig_v.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,46,0.3)')
        st.plotly_chart(fig_v, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="glass-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
            <div style="color:#9ca3af;font-size:1.1rem;">Click <strong>Run Simulation</strong> to generate S-parameter analysis</div>
        </div>
        ''', unsafe_allow_html=True)

# ==================== TAB 3: YIELD ANALYSIS ====================
with tab3:
    st.markdown('<div class="section-header">🎲 Manufacturing Yield Analysis (Monte Carlo)</div>', unsafe_allow_html=True)

    st.markdown("""
    Estimate production yield by simulating manufacturing tolerances:
    - **Substrate εr & Height**: ±5% variation
    - **Etching Tolerance**: ±20µm width variation
    """)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        num_runs = st.slider("Number of Runs", 50, 500, 100)
    with col2:
        tolerance = st.slider("Substrate Tolerance (%)", 1, 10, 5) / 100.0
    
    if st.button("🎲 Run Monte Carlo Analysis", type="primary"):
        with st.spinner(f"Simulating {num_runs} manufacturing variations..."):
            monte_carlo_res = simulator.run_monte_carlo(num_runs, tolerance)
            st.session_state['monte_carlo'] = monte_carlo_res
        st.success("✅ Analysis complete!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'monte_carlo' in st.session_state:
        res = st.session_state['monte_carlo']
        yield_pct = res['yield']
        
        color = "#10b981" if yield_pct > 80 else "#f59e0b" if yield_pct > 50 else "#ef4444"
        
        st.markdown(f'''
        <div class="glass-card" style="text-align:center; padding: 2rem;">
            <div style="font-size:1.2rem;color:#9ca3af;">Expected Production Yield</div>
            <div style="font-size:4rem;font-weight:800;color:{color};">{yield_pct:.1f}%</div>
            <div style="color:#9ca3af;">{int(yield_pct * res['runs'] / 100)} passed out of {res['runs']} runs</div>
        </div>
        ''', unsafe_allow_html=True)

# ==================== TAB 4: COMPARISON ====================
with tab4:
    st.markdown('<div class="section-header">⚖️ Conventional vs Wideband Design</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
    run_comp = st.button("▶️ Run Comparison Analysis", type="primary", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if run_comp:
        with st.spinner("🔄 Analyzing both configurations..."):
            comparison = simulator.run_comparison()
            st.session_state['comparison'] = comparison
        st.success("✅ Comparison complete!")
    
    if 'comparison' in st.session_state:
        comparison = st.session_state['comparison']
        single = comparison['single_stage']
        multi = comparison['multistage']
        improvement = comparison['improvement']
        
        # Improvement summary
        st.markdown('<div class="section-header">🚀 Improvements Achieved</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            delta = improvement['bandwidth_improvement_percent']
            color = "#10b981" if delta > 0 else "#ef4444"
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:{color};">+{delta:.1f}%</div>
                <div style="color:#9ca3af;">Bandwidth Improvement</div>
            </div>
            ''', unsafe_allow_html=True)
        with c2:
            delta = improvement['return_loss_improvement_db']
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#3b82f6;">{delta:.1f} dB</div>
                <div style="color:#9ca3af;">Return Loss Improvement</div>
            </div>
            ''', unsafe_allow_html=True)
        with c3:
            delta = improvement['phase_improvement_deg']
            st.markdown(f'''
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:700;color:#f59e0b;">{delta:.1f}°</div>
                <div style="color:#9ca3af;">Phase Balance Improvement</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Comparison plots
        st.markdown('<div class="section-header">📊 Side-by-Side Comparison</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_c1 = plot_comparison(single['simulation'], multi['simulation'], 's11_mag_db', 'Return Loss (S11)')
            fig_c1.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,46,0.3)')
            st.plotly_chart(fig_c1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_c2 = plot_comparison(single['simulation'], multi['simulation'], 'phase_difference_deg', 'Phase Difference')
            fig_c2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,46,0.3)')
            st.plotly_chart(fig_c2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed comparison table
        st.markdown('<div class="section-header">📋 Detailed Metrics Comparison</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="glass-card">
        <table class="data-table">
            <tr>
                <th>Metric</th>
                <th>Conventional (1-Stage)</th>
                <th>Wideband (2-Stage)</th>
                <th>Improvement</th>
            </tr>
            <tr>
                <td>Bandwidth</td>
                <td>''' + f'{single["metrics"]["bandwidth_percent"]:.1f}%' + '''</td>
                <td><strong>''' + f'{multi["metrics"]["bandwidth_percent"]:.1f}%' + '''</strong></td>
                <td style="color:#10b981;">+''' + f'{improvement["bandwidth_improvement_percent"]:.1f}%' + '''</td>
            </tr>
            <tr>
                <td>Return Loss</td>
                <td>''' + f'{single["metrics"]["min_return_loss_db"]:.1f} dB' + '''</td>
                <td><strong>''' + f'{multi["metrics"]["min_return_loss_db"]:.1f} dB' + '''</strong></td>
                <td style="color:#3b82f6;">''' + f'{improvement["return_loss_improvement_db"]:.1f} dB' + '''</td>
            </tr>
            <tr>
                <td>Phase Imbalance</td>
                <td>''' + f'{single["metrics"]["max_phase_imbalance_deg"]:.1f}°' + '''</td>
                <td><strong>''' + f'{multi["metrics"]["max_phase_imbalance_deg"]:.1f}°' + '''</strong></td>
                <td style="color:#f59e0b;">''' + f'{improvement["phase_improvement_deg"]:.1f}°' + '''</td>
            </tr>
            <tr>
                <td>Max VSWR</td>
                <td>''' + f'{single["metrics"]["max_vswr"]:.2f}' + '''</td>
                <td><strong>''' + f'{multi["metrics"]["max_vswr"]:.2f}' + '''</strong></td>
                <td>—</td>
            </tr>
        </table>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="glass-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">⚖️</div>
            <div style="color:#9ca3af;font-size:1.1rem;">Click <strong>Run Comparison</strong> to compare conventional and wideband designs</div>
        </div>
        ''', unsafe_allow_html=True)

# ==================== TAB 5: REPORT ====================
with tab5:
    st.markdown('<div class="section-header">📋 Design Summary Report</div>', unsafe_allow_html=True)
    
    dimensions = microstrip.branchline_dimensions(center_freq)
    
    st.markdown(f'''
    <div class="glass-card">
        <h4 style="color:#667eea;margin:0 0 1rem 0;">🔧 Design Specifications</h4>
        <table class="data-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Center Frequency</td><td><strong>{center_freq} GHz</strong></td></tr>
            <tr><td>Target Bandwidth</td><td>≥ {bandwidth}%</td></tr>
            <tr><td>Characteristic Impedance</td><td>{z0} Ω</td></tr>
            <tr><td>Substrate</td><td>{substrate_choice}</td></tr>
            <tr><td>Dielectric Constant (εr)</td><td>{epsilon_r}</td></tr>
            <tr><td>Substrate Thickness</td><td>{thickness} mm</td></tr>
            <tr><td>Loss Tangent (tan δ)</td><td>{loss_tan}</td></tr>
            <tr><td>Number of Stages</td><td>{num_stages}</td></tr>
        </table>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="glass-card">
        <h4 style="color:#10b981;margin:0 0 1rem 0;">📏 Calculated Dimensions</h4>
        <table class="data-table">
            <tr><th>Component</th><th>Impedance</th><th>Width</th><th>Length (λ/4)</th></tr>
            <tr>
                <td>Series Arms</td>
                <td>{dimensions['series_arm']['impedance']:.2f} Ω</td>
                <td><strong>{dimensions['series_arm']['width_mm']:.3f} mm</strong></td>
                <td><strong>{dimensions['series_arm']['quarter_wave_mm']:.2f} mm</strong></td>
            </tr>
            <tr>
                <td>Shunt Arms</td>
                <td>{dimensions['shunt_arm']['impedance']:.2f} Ω</td>
                <td><strong>{dimensions['shunt_arm']['width_mm']:.3f} mm</strong></td>
                <td><strong>{dimensions['shunt_arm']['quarter_wave_mm']:.2f} mm</strong></td>
            </tr>
        </table>
    </div>
    ''', unsafe_allow_html=True)
    
    if 'sim_metrics' in st.session_state:
        metrics = st.session_state['sim_metrics']
        st.markdown(f'''
        <div class="glass-card">
            <h4 style="color:#3b82f6;margin:0 0 1rem 0;">📊 Simulation Results</h4>
            <table class="data-table">
                <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
                <tr>
                    <td>Bandwidth</td>
                    <td>{metrics['bandwidth_percent']:.1f}%</td>
                    <td>≥ {bandwidth}%</td>
                    <td>{"✅" if metrics['bandwidth_percent'] >= bandwidth else "⚠️"}</td>
                </tr>
                <tr>
                    <td>Return Loss</td>
                    <td>{metrics['min_return_loss_db']:.1f} dB</td>
                    <td>< -10 dB</td>
                    <td>{"✅" if metrics['min_return_loss_db'] < -10 else "⚠️"}</td>
                </tr>
                <tr>
                    <td>Max VSWR</td>
                    <td>{metrics['max_vswr']:.2f}</td>
                    <td>< 2</td>
                    <td>{"✅" if metrics['max_vswr'] < 2 else "⚠️"}</td>
                </tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="glass-card" style="border-left:4px solid #f59e0b;">
        <h4 style="color:#f59e0b;margin:0 0 1rem 0;">⚠️ Limitations & Notes</h4>
        <ul style="color:#9ca3af;margin:0;padding-left:1.5rem;">
            <li>Simulation-based results only - fabrication recommended for validation</li>
            <li>FR4 substrate has higher loss at frequencies above 3 GHz</li>
            <li>Multistage designs require larger PCB area</li>
            <li>Manufacturing tolerances may affect actual performance</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 6: THEORY ====================
with tab6:
    st.markdown('<div class="section-header">📚 Theory & Background</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#667eea;margin:0 0 1rem 0;">What is a Balun?</h4>
            <p style="color:#d1d5db;">A <strong>Balun</strong> (Balanced-to-Unbalanced) is an RF component that:</p>
            <ul style="color:#9ca3af;">
                <li>Converts single-ended to differential signals</li>
                <li>Provides 180° phase difference between outputs</li>
                <li>Maintains equal amplitude at both outputs</li>
                <li>Used in mixers, amplifiers, and antenna feeds</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#10b981;margin:0 0 1rem 0;">Why Multistage?</h4>
            <ul style="color:#9ca3af;">
                <li>Single-stage: ~10% bandwidth</li>
                <li>Multistage: 25%+ bandwidth achievable</li>
                <li>Gradual impedance matching</li>
                <li>Better phase stability over frequency</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="glass-card">
            <h4 style="color:#f59e0b;margin:0 0 1rem 0;">Key Formulas</h4>
            <div class="formula-box">
                <strong>Free-space wavelength:</strong><br>
                λ₀ = c / f
            </div>
            <div class="formula-box">
                <strong>Effective εr:</strong><br>
                εeff = (εr+1)/2 + (εr-1)/2 × (1+12h/W)^-0.5
            </div>
            <div class="formula-box">
                <strong>Guided wavelength:</strong><br>
                λg = λ₀ / √εeff
            </div>
            <div class="formula-box">
                <strong>VSWR:</strong><br>
                VSWR = (1 + |Γ|) / (1 - |Γ|)
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="glass-card">
        <h4 style="color:#ec4899;margin:0 0 1rem 0;">Branchline Topology</h4>
        <pre style="color:#d1d5db;background:rgba(0,0,0,0.3);padding:1rem;border-radius:8px;font-family:'JetBrains Mono',monospace;">
         Port 2 (Output 1, 0°)
           │
       ┌───┴───┐
       │  λ/4  │  ← Series Arms (Z₀/√2 ≈ 35.35Ω)
Port 1 ┤       ├ Port 4 (Isolated)
       │  λ/4  │  ← Shunt Arms (Z₀ = 50Ω)  
       └───┬───┘
           │
         Port 3 (Output 2, 180°)
        </pre>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 7: CAD EXPORT ====================
with tab7:
    st.markdown('<div class="section-header">💾 CAD Layout Export</div>', unsafe_allow_html=True)
    
    # Initialize Generator
    from core.layout_generator import LayoutGenerator
    from components.visualizations import plot_high_fidelity_layout
    
    layout_gen = LayoutGenerator({
        'epsilon_r': epsilon_r,
        'thickness_mm': thickness
    })
    
    # Generate layout data
    dimensions = microstrip.branchline_dimensions(center_freq)
    layout_data = layout_gen.generate_layout(dimensions, num_stages)
    
    col_cad1, col_cad2 = st.columns([2, 1])
    
    with col_cad1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📐 High-Fidelity Interactive Viewer")
        
        # Plot precise layout
        shapes = layout_gen.get_plotly_shapes(layout_data)
        fig_layout = plot_high_fidelity_layout(shapes, f"{num_stages}-Stage Wideband Balun Layout")
        st.plotly_chart(fig_layout, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_cad2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Export Options")
        
        st.info("Export industry-standard files for fabrication or full-wave EM simulation.")
        
        # Prepare file buffers
        import io
        import os
        
        # GDSII Export
        # Save to temp file then read bytes
        gds_filename = f"balun_{num_stages}stage_{center_freq}GHz.gds"
        layout_gen.export_gds(layout_data, gds_filename)
        with open(gds_filename, "rb") as f:
            gds_bytes = f.read()
        os.remove(gds_filename)
        
        st.download_button(
            label="💾 Download GDSII (.gds)",
            data=gds_bytes,
            file_name=gds_filename,
            mime="application/octet-stream",
            use_container_width=True,
            help="Standard format for Professional EM Tools (ADS, HFSS, Sonnet)"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # DXF Export
        dxf_filename = f"balun_{num_stages}stage_{center_freq}GHz.dxf"
        layout_gen.export_dxf(layout_data, dxf_filename)
        with open(dxf_filename, "rb") as f:
            dxf_bytes = f.read()
        os.remove(dxf_filename)
        
        st.download_button(
            label="📐 Download DXF (.dxf)",
            data=dxf_bytes,
            file_name=dxf_filename,
            mime="application/dxf",
            use_container_width=True,
            help="Format for Mechanical CAD (AutoCAD, SolidWorks)"
        )
        
        st.markdown("---")
        st.markdown("#### 📋 Layout Stats")
        bounds = layout_data['bounds']
        total_w = bounds['max_x'] - bounds['min_x']
        total_h = bounds['max_y'] - bounds['min_y']
        
        st.markdown(f"""
        **Total Area:** {total_w/1000:.1f} x {total_h/1000:.1f} mm  
        **Layer:** 1 (Trace), 10 (Labels)  
        **Units:** Micro-meters (µm)  
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
<div class="footer">
    <p>📡 <strong>Wideband Branchline Balun Designer</strong></p>
    <p>Professional RF Design Tool • Simulation-based validation</p>
</div>
''', unsafe_allow_html=True)
