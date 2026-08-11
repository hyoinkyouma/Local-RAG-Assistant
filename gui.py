import os
import sys
import threading

import webview
from path_utils import RES_DIR

HOST = "127.0.0.1"
PORT = 8000
APP_URL = f"http://{HOST}:{PORT}"
LOADING_URL = "file://" + os.path.join(RES_DIR, "static", "loading.html").replace("\\", "/")


def start_server():
    import uvicorn
    from server import app
    os.chdir(RES_DIR)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def on_closing():
    os._exit(0)


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    window = webview.create_window(
        "DocuStore Local Assistant",
        LOADING_URL,
        width=1280,
        height=800,
        resizable=True,
        min_size=(800, 600),
    )
    window.events.closing += on_closing
    webview.start()
