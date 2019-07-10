import os
import sys
from pathlib import Path

cur = Path.cwd()
while cur.name != "src":
    cur = root.parent

prev = cur.parent
if not cur in sys.path:
    sys.path.append(str(cur))

from utils.parse_config import Config
from utils.bridge import CBridge

if __name__ == "__main__":
    # Initialize configurations
    # -------------------------
    config = Config(prev.joinpath("conf", "neuzz.yml"))
    config = config.load_config()

    # Create a comm bridge to interact with the C code
    # ------------------------------------------------
    with CBridge(config.socket.HOST, config.socket.PORT, verbose=True) as conn:
        # Generate Gradients
        gen_gradients('train')
        # Ask the C code to take over and generate inputs for fuzzing
        conn.sendall("start")
        data = conn.recv(config.socket.RECV_BUFF)
        while data:
            generate_gradients(data)
            conn.sendall("start")
