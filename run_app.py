import os
import subprocess

if __name__ == "__main__":
    # run from project root
    subprocess.run(["streamlit", "run", "src/ui/app.py"], check=True)
