import sys
from subprocess import call

#if not (sys.executable).endswith(("python","python3","python3.10")):
#    print("Sys failed to find environment executeable, exitting")
#    exit()

# See comments in setup.py for more info
call([sys.executable, "bobTheBuilder/setup.py", "develop"])