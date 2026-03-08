"""Launcher script for Email Privilege Dashboard."""
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the email privilege dashboard using Streamlit."""
    dashboard_path = Path(__file__).parent / "email_privilege_dashboard.py"
    project_root = Path(__file__).resolve().parents[1]
    
    print("Starting Email Privilege Dashboard...")
    print("The dashboard will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Run streamlit
    subprocess.run(
        [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.headless=false"
        ],
        cwd=str(project_root),
    )


if __name__ == "__main__":
    main()
