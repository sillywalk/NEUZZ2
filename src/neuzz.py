import os
import sys
from pathlib import Path

cur = Path.cwd()
while cur.name != "src":
    cur = cur.parent

prev = cur.parent
if not cur in sys.path:
    sys.path.append(str(cur))

from copy import copy, deepcopy
from utils.parse_config import Config
from utils.bridge import CBridge
from gradients import Gradient

if __name__ == "__main__":
    # -------------------------
    # Initialize configurations
    # -------------------------
    usr_config = Config(prev.joinpath("conf", "neuzz.yml"))
    usr_config = usr_config.load_config()
    # ------------------------------------------------
    # Create a comm bridge to interact with the C code
    # ------------------------------------------------
    with CBridge(config.socket.HOST, config.socket.PORT, verbose=True) as conn:
        # Generate Gradients
        grad = Gradient(usr_config)
        gen_gradients('train')
        # Ask the C code to take over and generate inputs for fuzzing
        conn.sendall("start")
        data = conn.recv(config.socket.RECV_BUFF)
        while data:
            generate_gradients(data)
            conn.sendall("start")
