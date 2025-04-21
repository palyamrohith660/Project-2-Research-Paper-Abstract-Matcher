import webview
import threading
import subprocess
import time

def run_streamlit():
    """Launch the Streamlit app."""
    subprocess.Popen(["streamlit", "run", "t1.py", "--server.port=8501", "--server.headless=true"])

def open_webview():
    """Wait for the localhost server and open the PyWebView window."""
    time.sleep(3)  # Delay to ensure Streamlit is up
    webview.create_window("Teachers Panel", "http://localhost:8501")

if __name__ == "__main__":
    threading.Thread(target=run_streamlit, daemon=True).start()
    open_webview()
