import manim

# If we attempted direct import, this would crash on first try and succeed on second
#from importlib.machinery import ExtensionFileLoader 
#canim = ExtensionFileLoader("canim", "canim.cpython-310-x86_64-linux-gnu.so").load_module()

import canim

print("tests started")

print(canim.canim_test())
print(canim.threanim_test())

testThing = manim.Polygon([1,0,0], [0,1,0], [0,0,0])
print(canim.FW_display_vectorized(testThing,testThing))

print("tests complteted")