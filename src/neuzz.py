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
from core.gradients import Gradient

if __name__ == "__main__":
    # -------------------------
    # Initialize configurations
    # -------------------------
    usr_config = Config(prev.joinpath("config", "neuzz.yml"))
    usr_config = usr_config.load_config()
    # ------------------------------------------------
    # Create a comm bridge to interact with the C code
    # ------------------------------------------------
    with CBridge(usr_config.socket.HOST, usr_config.socket.PORT, verbose=True) as conn:
        # Get system args
        arg_list = sys.argv[1:]
        # Generate Gradients
        grad = Gradient(config=usr_config, argv=arg_list, verbose=True)
        grad = grad.generate_gradients("train")
        # Ask the C code to take over and generate inputs for fuzzing
        conn.sendall("start")
        data = conn.recv(config.socket.RECV_BUFF)
        while data:
            grad = grad.generate_gradients(data)
            conn.sendall("start")
