from distutils.util import run_2to3
from math import atan2
from multiprocessing.dummy import current_process
from re import A
from manim import *
from manim.utils.paths import path_along_circles
import numpy as np
from sharedTools import *


intermediateTop = lambda alpha: 0 if alpha==0 else 1
earlyFinish = lambda finish, internal=smooth: (lambda alpha: internal(alpha/finish) if internal(alpha/finish)<=1 else 1)
finishLate = lambda finish, internal=smooth: (lambda alpha: internal(alpha)*finish )
headStart = lambda start, internal=smooth: (lambda alpha: internal(alpha * start) / internal(start) )
doubleHeadStart = lambda start, internal=smooth: ((lambda alpha: internal(start * alpha + 1 - 1 / start) - internal(1-start)) / internal(start) )
reverse = lambda func: lambda t: 1-func(t)

class dropFade(Transform):
    def __init__(self, mobject, target, **kwargs):
        super().__init__(mobject, target, **kwargs)
        self.final_target = target.copy()
        self.mobject.set_fill(opacity=0)
        self.mobject.set_stroke(opacity=0)
        self.mobject.set_background_stroke(opacity=0)

    def create_starting_mobject(self) -> Mobject:
        temp = self.mobject.copy()
        #temp = temp.scale(0.6)
        temp = temp.shift(UP)
        return temp

class OpacReverseTransform(Transform):
    def __init__(self, mobject: Mobject, target_mobject: Mobject, opac_rate_func = smooth, **kwargs) -> None:
        super().__init__(target_mobject, mobject, **kwargs)
        self.opac_rate_func = opac_rate_func

    def interpolate_submobject(self, submobject: Mobject, starting_submobject: Mobject, target_copy: Mobject, alpha: float) -> "Transform":
        submobject.points = self.path_func(starting_submobject.points, target_copy.points, 1-alpha)
        submobject.interpolate_color(starting_submobject, target_copy, 1-self.opac_rate_func(alpha))

class ColorControlTransform(Transform):
    def __init__(self, mobject: Mobject, target_mobject: Mobject, color_rate_func = smooth, **kwargs) -> None:
        super().__init__(mobject, target_mobject, **kwargs)
        self.color_rate_func = color_rate_func
        self.true_rate_func = self.rate_func
        self.rate_func = linear

    def interpolate_submobject(self, submobject: Mobject, starting_submobject: Mobject, target_copy: Mobject, alpha: float) -> "Transform":
        submobject.points = self.path_func(starting_submobject.points, target_copy.points, self.true_rate_func(alpha))
        submobject.interpolate_color(starting_submobject, target_copy,self.color_rate_func(alpha))

class dropInFromTo(Transform):
    def __init__(self, mobject: Mobject, comefrom=[0,0,0], **kwargs):
        self.comefrom = comefrom
        super().__init__(mobject,**kwargs)
    
    def create_starting_mobject(self) -> Mobject:
        t= self.mobject.copy()
        t.move_to(self.comefrom)
        return t


class introSequence(Scene):
    def construct(self):
        self.performIntroSequence()

    def performIntroSequence(self):
        #This function generates the triangles for the hexagon in the intro vid, i is the triangle number, j is the point on it
        f,g = map(lambda func: (lambda t1,cos30:(lambda i,j,r1=0.6,r2=3: r1*func(np.pi/12 + t1*(i-1)) +
            (0 if j == 1 else (
                (   r2 * func(t1*(i-1))    if i%2==1 else cos30 * r2 * func(t1*(i-1))   ) if j==2 else
                (   r2 * func(t1*i)        if i%2==0 else cos30 * r2 * func(t1*i)     )
        ))))(np.pi/6,np.sqrt(3)/2),[np.cos,np.sin])

        logo = [Polygon(*map(lambda j: [f(i,j),g(i,j),0],[0,1,2]),color=WHITE) for i in range(12)]
        [x.set_fill(color=WHITE,opacity=1) for x in logo]
        logo = VGroup(*[logo[i] for i in [3,2,1,0,11,10,9,8,7,6,5,4]])

        step1 = AnimationGroup(*[OpacReverseTransform(logo[-1].copy().scale((0.6)*(1-i/11) + i/11).set_fill(opacity=0).set_stroke(opacity=0),
        logo[i], path_func = path_along_circles(-2*PI * (11-i)/12, [0,0,0]),opac_rate_func=earlyFinish(0.2,smooth),
         rate_func=finishLate(0.7,internal=smooth) if i==11 else smooth, run_time= 0.7 if i==11 else 1) for i in range(12)],lag_ratio=0.2)

        logoBackUp = logo.copy()


        N10KYA=Tex("N10KYA").scale(1.5)[0]
        [N10KYA[i].move_to([-4,2.5-i,0]) for i in range(6)]

        dropLetters=AnimationGroup(*[TransformFromCopy(N10KYA[5-i].copy().move_to([-4,5,0]),N10KYA[5-i]) for i in range(6)],lag_ratio=0.2)

        self.play(AnimationGroup(step1,dropLetters,lag_ratio=0.5))


        ###################################################STEP1 COMPLETE


        logoIn = [Polygon(*map(lambda j: [f(i,j,r1=0),g(i,j,r1=0),0],[0,1,2]),color=WHITE) for i in range(12)]
        [x.set_fill(color=WHITE,opacity=1) for x in logoIn]
        [x.set_stroke(color=WHITE,width=x.get_stroke_width()*0.3) for x in logoIn]
        logoIn = VGroup(*[logoIn[i] for i in [3,2,1,0,11,10,9,8,7,6,5,4]])
        step2 = AnimationGroup(*[Transform(logo[i],logoIn[i]) for i in range(12)],run_time=0.4)

        hexag = Polygon(*[[3.3*np.cos(i*PI/3),3.3*np.sin(i*PI/3),0] for i in range(6)])
        hexag.round_corners(radius=0.2)
        hexag.set_stroke(color=WHITE,opacity=1,width=12)
        hexag.z_index = -1
        
        opacitylayers,hexsplit=(2*12,6)
        hexArmy = np.zeros((opacitylayers,hexsplit),dtype=object)
        for i in range(opacitylayers):
            for j in range(hexsplit):
                hexArmy[i,j]=hexag.copy()
                hexArmy[i,j].points = (lambda length,rotate: np.concatenate((rotate.points[int(j*length/hexsplit):], rotate.points[0:int(j*length/hexsplit)])))(len(hexag.points),hexArmy[i,j])
                if i%2==1: hexArmy[i,j].points = np.flip(hexArmy[i,j].points,axis=0)
                hexArmy[i,j].z_index= -1
                hexArmy[i,j].set_stroke(opacity=i/(opacitylayers-1))

        step2andhalf = AnimationGroup(*[AnimationGroup(*[Create(hexArmy[i,j],
        rate_func=finishLate(((hexsplit*2)+1)/((hexsplit*2)**2),smooth),run_time=1/3.)
         for i in range(opacitylayers)],lag_ratio=0.1) for j in range(hexsplit)])


        N10KYAFull=VGroup(*[Tex("NOT")[0],Tex("10 000")[0],Tex("YEARS")[0],Tex("AGO")[0]]).scale(1.5)
        for i in N10KYAFull:
            for j in i[1:]:
                j.shift((j.get_center() - i[0].get_center())*0.5)
        outlines = []
        for a in [(0,0),(1,1),(2,4),(3,5)]:
            dist = N10KYA[a[1]].get_center() - N10KYAFull[a[0]][0].get_center()
            N10KYAFull[a[0]].shift(dist)
            outlines+=[N10KYAFull[a[0]].copy().set_fill(opacity=0,color='#010101').set_stroke(opacity=1,color='#010101',width=10)]
            N10KYAFull[a[0]]= N10KYAFull[a[0]][1:]
        outlines=VGroup(*outlines)
        for a in outlines.family_members_with_points():
            a.z_index=1
            #a.scale(1.2)
        for a in N10KYAFull.family_members_with_points():
            a.z_index=2
        for a in N10KYA:
            a.z_index=2
            
        a=[Create(a,run_time=1.2) for a in outlines.family_members_with_points()]
        b=[Write(a,run_time=1.2) for a in N10KYAFull.family_members_with_points()]
        wordAnimation = AnimationGroup(FadeOut(N10KYA[2:4],run_time=0.3),AnimationGroup(*a,*b))

        self.play(AnimationGroup(AnimationGroup(step2,step2andhalf,lag_ratio=0.4),wordAnimation,lag_ratio=0.2))
        self.add(outlines)

        self.add(hexag)
        self.remove(*[hexArmy[int(i%opacitylayers),int(np.floor(i/opacitylayers))] for i in range(opacitylayers * hexsplit)])
        
        ###################################################STEP2 COMPLETE

        def moveAndRotateTri(triangle,addAngle,radius,i):
            currentAngle = i* PI /6 + PI/12
            a= triangle.move_to( radius * posFromAngle(currentAngle + addAngle) )
            return a.rotate(addAngle)

        def specialPathFunction(start_points: np.ndarray, end_points: np.ndarray, alpha: float):
            length = len(end_points)
            center_start = np.mean(start_points,axis=0)
            center_end = np.mean(end_points,axis=0)

            center_start_polar = np.array([angleFromPos(center_start),np.linalg.norm(center_start)])
            center_end_polar = np.array([angleFromPos(center_end),np.linalg.norm(center_end)])

            start_points_diff = start_points - center_start
            end_points_diff = end_points - center_end

            start_points_diff_polar = np.transpose(np.array([[angleFromPos(i) for i in start_points_diff],[np.linalg.norm(i) for i in start_points_diff]]))
            end_points_diff_polar = np.transpose(np.array([[angleFromPos(i) for i in end_points_diff],[np.linalg.norm(i) for i in end_points_diff]]))

            diff = center_end_polar[0] - center_start_polar[0]
            if abs(diff) > PI:
                center_end_polar[0] -= 2* PI * np.sign(diff)
#
            diffs = end_points_diff_polar[:,0] - start_points_diff_polar[:,0]
            for i in range(len(diffs)):
                if abs(diffs[i]) > PI: end_points_diff_polar[i,0] -= 2* PI * np.sign(diffs[i])
            
            center_polar = alpha * center_end_polar + (1-alpha) * center_start_polar
            points_diff_polar = alpha * end_points_diff_polar + (1-alpha) * start_points_diff_polar

            center = center_polar[1] * posFromAngle(center_polar[0])
            points_diff = [points_diff_polar[i,1] * posFromAngle(points_diff_polar[:,0])[i,:] for i in range(len(points_diff_polar[:,1]))]

            return center + points_diff

        specialRateFunc = lambda alpha: (smooth(alpha * 8.9) / smooth(0.89) )/10 if alpha<0.1 else 0.9 * ((4/5)*(alpha-0.1)**5 + (alpha-0.1) / 5) + 0.1
        step3AndHalf = Transform(hexag,hexag.copy().scale(5.4),run_time=3,rate_func = specialRateFunc)
        logo = VGroup(*[logo[i] for i in [2,1,0,11,10,9,8,7,6,5,4,3]])

        positions = [2,8,4,10,6,0,7,3,9,5,1,11]
        invPos = [ 5, 10,  0,  7,  2,  9,  4,  6,  1,  8,  3, 11]
        logoBit1 = AnimationGroup(*[ColorControlTransform(logo[i],moveAndRotateTri(logo[i].copy(),PI/6,14,i).set_fill(color=colors[invPos[i]]).set_stroke(color=colors[invPos[i]])
            ,path_func = specialPathFunction, run_time=3, color_rate_func = 
            (lambda f,j: (lambda t: np.power(np.sin(4*PI*t + f * 3 * PI / 4),2) if 
            (invPos[j])/16 < t < (invPos[j] + 4)/16 else 0))(invPos[i]%4,i)
             ,rate_func=specialRateFunc) for i in range(12)])

        a=[Uncreate(a,run_time=1) for a in outlines.family_members_with_points()]
        b=[Unwrite(a,run_time=1) for a in N10KYAFull.family_members_with_points()+list((*N10KYA[0:2],*N10KYA[4:6]))]

        self.play(AnimationGroup(AnimationGroup(logoBit1,step3AndHalf),AnimationGroup(*a,*b),lag_ratio=0.8))

