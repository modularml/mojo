import shutil
import subprocess
from importlib.util import find_spec

FIX = """
-------------------------------------------------------------------------
fix following the steps here:
    https://github.com/modularml/mojo/issues/1085#issuecomment-1771403719
-------------------------------------------------------------------------
"""


def install_if_missing(name: str):
    if find_spec(name):
        return

    print(f"{name} not found, installing...")
    try:
        if shutil.which("python3"):
            python = "python3"
        elif shutil.which("python"):
            python = "python"
        else:
            raise ImportError("python not on path" + FIX)
        subprocess.check_call([python, "-m", "pip", "install", name])
        return
    except:
        raise ImportError(f"{name} not found" + FIX)
