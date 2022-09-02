import sys
from subprocess import call

if not (sys.executable).endswith(("python","python3","python3.10")):
    print("Sys failed to find environment executeable, exitting")
    exit()

# See comments in setup.py for more info
call([sys.executable, "bobTheBuilder/setup.py", "develop"])

# If we attempted direct import, this would crash on first try and succeed on second
from importlib.machinery import ExtensionFileLoader 
canim = ExtensionFileLoader("canim", "canim.cpython-310-x86_64-linux-gnu.so").load_module()
print(canim.canim_test())
print(canim.danim_test()) 
