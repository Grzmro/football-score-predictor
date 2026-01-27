import os
import subprocess
import sys

def main():
    """glowny skrypt do odpalania aplikacji streamlit"""
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'gui', 'streamlit_app.py')
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    print(f"Uruchamianie aplikacji Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
