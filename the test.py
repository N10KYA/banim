from manim import *
from banim import myCamera
from time import time
from intro import introSequence


class normalScene(Scene):
    def construct(self):
        introSequence.performIntroSequence(self)

scene1 = normalScene()
scene2 = introSequence(camera_class= myCamera)

#time1 = time()
#with tempconfig({   "quality": "high_quality"}):
#    scene1.render()
#print("Took " + str(time() - time1) + " to do unparallelised version\n\n")

time2 = time()
with tempconfig({   "quality": "high_quality"}):
    scene2.render()
print("Took " + str(time() - time2) + " to do parallelised version\n\n")