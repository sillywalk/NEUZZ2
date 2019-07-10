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

if __name__ == "__main__":
    # Initialize configurations
    # -------------------------
    config = Config(prev.joinpath("conf", "neuzz.yml"))
    config = config.load_config()
