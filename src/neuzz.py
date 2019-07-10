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
    HOST = '127.0.0.1'
    PORT = 12012
    with CBridge(HOST, POST, verbose=True) as conn:
