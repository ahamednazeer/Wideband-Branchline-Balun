"""
Run script for Wideband Branchline Balun Designer
"""
import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    print("🚀 Starting Wideband Branchline Balun Designer...")
    print("📡 Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
