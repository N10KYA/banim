from __future__ import annotations
from calendar import month_name
from multiprocessing.sharedctypes import Value
#from cProfile import run
from typing import List, Tuple, Dict,\
    TYPE_CHECKING, Callable, Sequence

from manim import *
from manim.utils.rate_functions import *

from manim.mobject.opengl.opengl_mobject import OpenGLGroup

from manim.animation.animation import prepare_animation

from manim.utils.iterables import list_update, remove_list_redundancies

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup

import os
from scipy.interpolate import interp1d
from functools import partial

from manim.utils.hashing import _CustomEncoder, _Memoizer, KEYS_TO_FILTER_OUT
import zlib, json

NoneType = type(None)

posFromAngle = lambda t: np.array([np.cos(t),np.sin(t),0]) \
    if type(t) in [float, np.float64] \
    else np.transpose( \
        np.append( np.array([np.cos(t),np.sin(t)]),
            np.zeros((1,len(t))),
            axis=0) 
    )
angleFromPos = lambda t: np.arctan2(*np.flip(t[0:2]))

colors =[  '#ff0000','#00ffff','#9000ff','#00ff00',
            '#ffd000','#ff8800','#ff00ff','#00ff91',
            '#0000ff','#b4ff00','#00aaff','#ff00aa'
        ]

finishLate = lambda finish, internal=smooth: (lambda alpha: internal(alpha)*finish )

inverseSmooth = (lambda e: lambda t, i=10.: (2 + i + e(i)*i + 2 * np.log(t*(e(i)-1)/(1+e(i)+t-e(i)*t)) +
                 2 * e(i) * np.log(t*(e(i)-1)/(1+e(i)+t-e(i)*t)))/(2*i*(1+e(i))))(lambda t: np.exp(t/2))

# list of prime numbers to be used in random stuff
p=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
  223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
   347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
    463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]

pf = 999999937 #this prime number will be used to modulus big things, chosen to be under a billion
#               cuz 32 bits gives you about 10^9.6 so just over a billion

class Seed:
    seedHistory = [1]
    globalSeed: int = 1
    randomHistory = []

    def __init__(self, initialSeed: int = 1, seedHistoryToInherit: list[int] = [1],
                randomHistoryToInherit: list[float | tuple[float]] = [],
                relevantPrimes: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
                139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
                347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
                449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541],
                biggestNumAllowed = 999999937
        ):
        self.seedHistory = seedHistoryToInherit
        self.randomHistory = randomHistoryToInherit
        self.relevantPrimes = relevantPrimes
        self.biggestNumAllowed = biggestNumAllowed
        self.globalSeed = initialSeed

    def rand(self,change: int | NoneType, *args, **kwargs):
        np.random.seed(self.getSeed(change))
        result = np.random.random(*args,**kwargs)
        self.randomHistory += [result]
        return result

    def getSeed(self,change: int | NoneType):
        if change==None:
            return self.globalSeed
        self.globalSeed = (self.globalSeed * p[change%100])%pf
        self.seedHistory += [self.globalSeed]
        return self.globalSeed

    def setSeed(self,toSetTo):
        if toSetTo == 0:
            raise ValueError("Global Seed must never be set to 0 or else it cannot be updated")
        self.globalSeed = toSetTo
        self.seedHistory += [self.globalSeed]

    def specialRand(self, func: str,change: int | NoneType, *args,**kwargs):
        np.random.seed(self.getSeed(change))
        result = getattr(np.random,func)(*args,**kwargs)
        self.randomHistory += [result]
        return result


###########################################################################################################
############################################### Animation #################################################
###########################################################################################################

class CreateBothSides(Create):
    def _get_bounds(self, alpha: float):
        return (0.5-alpha/2,0.5+alpha/2)

class CreateCustom(Create):
    def __init__(self, mobject, bounds_func = None,
         lag_ratio: float = 1, introducer: bool = True, **kwargs) -> None:
        super().__init__(mobject, lag_ratio, introducer, **kwargs)
        if bounds_func==None:
            self.bounds_func = lambda t: (0,t)
        else:
            self.bounds_func = bounds_func
    
    def _get_bounds(self, alpha: float):
        return self.bounds_func(alpha)

class CreateTransform(Transform):
    def __init__(self, mobject, target, **kwargs):
        super().__init__(mobject, target, **kwargs)
        self.final_target = target.copy()
        self.mobject.set_fill(opacity=0)
        self.mobject.set_stroke(opacity=0)
        self.mobject.set_background_stroke(opacity=0)

    def interpolate_submobject(self,
            submobject: Mobject,
            starting_submobject: Mobject,
            target_copy: Mobject,
            alpha: float
            ) -> "Transform":
        submobject.points = self.path_func(
            starting_submobject.points, target_copy.points, alpha
        )
        submobject.interpolate_color(
            starting_submobject, target_copy, 0 if alpha==0 else 1)

class ReplaceFadeOut(FadeOut):
    def interpolate_submobject(self, submobject: Mobject, starting_submobject: Mobject,
        target_copy: Mobject, alpha: float):
        submobject.interpolate_color(starting_submobject, target_copy, 1 if alpha==0 else alpha)

class OpacTransform(Transform):
    def __init__(self, mobject: Mobject, target_mobject: Mobject, opac_rate_func = smooth, **kwargs) -> None:
        super().__init__(mobject,target_mobject, **kwargs)
        self.opac_rate_func = opac_rate_func
        self.true_rate_func = self.rate_func
        self.rate_func = linear

    def interpolate_submobject(self, submobject: Mobject, starting_submobject: Mobject, target_copy: Mobject, alpha: float) -> "Transform":
        submobject.points = self.path_func(starting_submobject.points, target_copy.points, self.true_rate_func(alpha))
        submobject.interpolate_color(starting_submobject, target_copy, self.opac_rate_func(alpha))

class AnimationGestalt(AnimationGroup):
    def __init__(
        self, *animations: Animation, run_time: float = 1, rate_func: Callable[[float], float] = linear,
         lag_ratio: float = 0, **kwargs ) -> None:
        newAnimations = []
        self.animations = [prepare_animation(anim) for anim in animations]
        self.lag_ratio = lag_ratio

        timing = self.rebuild_animations_with_timings()

        total_time = timing[-1][2]
        anims_with_alphas = [(a[0],a[1]/total_time,a[2]/total_time) for a in timing]
        for i in anims_with_alphas:
            newAnimations += [
                i[0].set_rate_func((lambda g,h,a,b: lambda t2:
                g( 
                    (lambda t1: (t1-a)/(b-a) if a<=t1<=b else (0 if a>t1 else 1))( h(t2) ) 
                )
                )(i[0].rate_func,rate_func,i[1],i[2])
                ).set_run_time(run_time)
            ]

        super().__init__(*newAnimations, run_time=run_time, rate_func=linear, lag_ratio=0, **kwargs)

    def rebuild_animations_with_timings(self) -> None:
        """
        like build_animations_with_timings but instead of working for the renderer,
        it works for me and figures out how to reassign animation properties
        """
        anims_with_timings = []
        curr_time: float = 0
        for anim in self.animations:
            start_time: float = curr_time
            end_time: float = start_time + anim.get_run_time()
            anims_with_timings.append((anim, start_time, end_time))
            curr_time = (1 - self.lag_ratio) * start_time + self.lag_ratio * end_time
        return anims_with_timings


##########################################################################
## SUCCESSIONPRECISION RUNS EXTREMELY SLOWLY, DO NOT USE UNTIL REWORKED ##
##########################################################################
class SuccessionPrecision(AnimationGroup):
    """
    Instead of taking animations, it takes a tuple with an animation,
    its desired start time, and its desired duration. If desired
    duration is not included, it will inherit the run_time from
    the animation argument. If nothing except the animation is included,
    it will assume start time to be zero and inherit duration.
    """
    def __init__(self, *animationsWithTimings: tuple(Animation,float,float) |
        list(Animation,float,float) | tuple(Animation,float) |
        list(Animation,float) | Animation,
        group: Group | VGroup | OpenGLGroup | OpenGLVGroup = None,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = ...,
        lag_ratio: float = 0,
        delayed_start: float = 0,
        switchDurationToLength = False,
         **kwargs) -> None:

        n=len(animationsWithTimings)

        adjustableCopy = list(animationsWithTimings)
        for i in range(n):
            if isinstance(animationsWithTimings[i],Animation):
                adjustableCopy[i] = list((animationsWithTimings[i],0,animationsWithTimings[i].run_time))
            else:
                if len(animationsWithTimings[i]) == 2:
                    adjustableCopy[i] = list((*adjustableCopy[i],animationsWithTimings[i][0].run_time))
                else:
                    adjustableCopy[i] = list(adjustableCopy[i])
        self.anims_with_timings = adjustableCopy
        for i in range(n):
            self.anims_with_timings[i][1] += delayed_start
        if not switchDurationToLength:
            for i in range(n):
                self.anims_with_timings[i][2] += self.anims_with_timings[i][1]
        self.activeAnimations = []

        if run_time==None:
            if n==1:
                self.run_time = self.anims_with_timings[0][2]
            else:
                self.run_time = max(*[self.anims_with_timings[-1][2] for i in range(n)])
        else:
            self.run_time = run_time

        for i in range(n):
            self.anims_with_timings[i][1] /= self.run_time
            self.anims_with_timings[i][2] /= self.run_time

        remember = self.anims_with_timings

        #forget
        super().__init__(*[self.anims_with_timings[i][0] for i in range(n)], group=group,
        run_time=self.run_time, rate_func=rate_func, lag_ratio=lag_ratio, **kwargs)

        self.anims_with_timings = remember

        #triggerpoints is 2*n x 2 in shape, each 3bit is trigger time, ID, 
        self.triggerPoints = (lambda t: np.append(t,np.zeros((2*n,1)),axis=1)    )(
            np.append(
                np.array([self.anims_with_timings[i][1] for i in range(n)]),
                np.array([self.anims_with_timings[i][2] for i in range(n)])).reshape((2*n,1))
        )

    
    def build_animations_with_timings(self) -> None:
        """This is managed as an argument"""
        pass
    
    def begin(self)-> None:
        print(self.group)
        assert len(self.animations) > 0
        self.update_active_animations(0)

    def finish(self) -> None:
        self.update_active_animations(1)

    def update_active_animations(self,alpha):
        for i in range(self.triggerPoints.shape[0]):
            if (not self.triggerPoints[i,1]) and alpha >= self.triggerPoints[i,0]:
                if i >= len(self.anims_with_timings):
                    idnum = i - len(self.anims_with_timings)
                    self.activeAnimations.remove(self.anims_with_timings[idnum])
                    self.anims_with_timings[idnum][0].finish()
                    self.triggerPoints[i,1] = 1
                else:
                    idnum = i
                    self.activeAnimations.append(self.anims_with_timings[idnum])
                    self.anims_with_timings[idnum][0]._setup_scene(self.sceneToRunSetupOn)
                    self.anims_with_timings[idnum][0].begin()
                    self.triggerPoints[i,1] = 1

    def interpolate(self,alpha: float) -> None:
        self.update_active_animations(alpha)
        for i in self.activeAnimations:
            i[0].interpolate((alpha - i[1])/(i[2] - i[1]))

    def _setup_scene(self, scene) -> None:
        """this is handled when things run begin"""
        self.sceneToRunSetupOn = scene

    def update_mobjects(self, dt: float) -> None:
            for i in self.activeAnimations:
                i[0].update_mobjects(dt)

    
###########################################################################################################
########################################### Better Versions ###############################################
###########################################################################################################          

def clearHouse():
    _Memoizer.reset_already_processed()

class BetterScene(Scene):
    """
    Attempt at improving Scene through anything I notice.
    Currently modifies the way mobjects are added when introducer animations
    run so that the animation can decide when to add the mobject.
    """
    def add_mobjects_from_animations(self, animations):
        def getRelevantSubMObjects(animations):
            thingsToAdd = []
            for animation in animations:
                if isinstance(animation,AnimationGroup):
                    thingsToAdd += getRelevantSubMObjects(animation.animations)
                elif animation.is_introducer():
                    pass
                else:
                    thingsToAdd += [animation.mobject]
        curr_mobjects = self.get_mobject_family_members()
        self.add(remove_list_redundancies([
            mobject for mobject in getRelevantSubMObjects(animations) if not mobject in curr_mobjects]))

class SortedDictCustomEncoder(_CustomEncoder):
    def _cleaned_iterable(self, iterable):
        def _key_to_hash(key):
            return zlib.crc32(json.dumps(key, cls=_CustomEncoder).encode())

        def _iter_check_list(lst):
            processed_list = [None] * len(lst)
            for i, el in enumerate(lst):
                el = _Memoizer.check_already_processed(el)
                if isinstance(el, (list, tuple)):
                    new_value = _iter_check_list(el)
                elif isinstance(el, dict):
                    new_value = _iter_check_dict(el)
                else:
                    new_value = el
                processed_list[i] = new_value
            return processed_list

        def _iter_check_dict(dct):
            processed_dict = {}
            items = list(dct.items())
            items.sort(key = lambda t: t[0])
            for k, v in items:
                v = _Memoizer.check_already_processed(v)
                if k in KEYS_TO_FILTER_OUT:
                    continue
                # We check if the k is of the right format (supporter by Json)
                if not isinstance(k, (str, int, float, bool)) and k is not None:
                    k_new = _key_to_hash(k)
                else:
                    k_new = k
                if isinstance(v, dict):
                    new_value = _iter_check_dict(v)
                elif isinstance(v, (list, tuple)):
                    new_value = _iter_check_list(v)
                else:
                    new_value = v
                processed_dict[k_new] = new_value
            return processed_dict

        if isinstance(iterable, (list, tuple)):
            return _iter_check_list(iterable)
        elif isinstance(iterable, dict):
            return _iter_check_dict(iterable)

def get_json(thing):
    stringify = json.dumps(thing, cls=SortedDictCustomEncoder)
    _Memoizer.reset_already_processed()
    return stringify

def hashMisc(misc: list | tuple, returnJsonToo: bool = False):
    stringify = [get_json(misc) for item in misc]
    if returnJsonToo:
        return ("_".join([str(zlib.crc32(repr(item).encode())) for item in stringify]), stringify)
    else:
        return "_".join([str(zlib.crc32(repr(item).encode())) for item in stringify])

###########################################################################################################
################################################ Mobject ##################################################
###########################################################################################################

class MyRectangle(Polygon):
    def __init__(self, width=1,height=1,subdiv=(1,1), rotate=0, **kwargs):
        subdiv = (int(subdiv[0]),int(subdiv[1]))
        rotate = int(rotate)
        vertices = [np.array([-width/2,-height/2,0])]
        #Start, end, number
        makeLine = (lambda s,e,n: list(np.transpose(
            np.array(list(map(lambda t: np.linspace(s[t],e[t],num=n),[0,1,2])))
        )))
        vertices += makeLine(
                        np.array([-width/2,-height/2,0]),
                        np.array([-width/2,height/2,0]),
                        subdiv[1]
                    )
        vertices += makeLine(
                        np.array([-width/2,height/2,0]),
                        np.array([width/2,height/2,0]),
                        subdiv[0]
                    )
        vertices += makeLine(
                        np.array([width/2,height/2,0]),
                        np.array([width/2,-height/2,0]),
                        subdiv[1]
                    )
        vertices += makeLine(
                        np.array([width/2,-height/2,0]),
                        np.array([-width/2,-height/2,0]),
                        subdiv[0]
                    )
        vertices = [vertices[(i + rotate) % len(vertices)] for i in range(len(vertices))]
        super().__init__(*vertices, **kwargs)

###########################################################################################################
############################################### Debugging #################################################
###########################################################################################################

def labelPoints(mobject, num: int) -> VGroup:
    """
    Returns label Tex MObjects on points
    """
    if type(mobject)!=list and type(mobject)!=np.ndarray:
        points = mobject.points
    else:
        points = mobject
    length = len(points)
    its = length / num
    nums = VGroup(*[Tex(str(i+1)) for i in range(num)])
    for i in range(num):
        nums[i].set_stroke(color='#ff0000',width=1).set_fill(color='#ff0000').scale(0.4)
        nums[i].move_to(points[int(its*i),:]).set_z_index(2)
    return nums


def detectBigDiff(mobject):
    """
    Takes a list of points and adds a 'noteable thing' if
    theres an unexpected change in distance.
    """
    normalDiff = np.linalg.norm(mobject.points[1,:] - mobject.points[0,:])
    noteableThings = []
    for i in range(len(mobject.points)-2):
        diff = np.linalg.norm(mobject.points[i+2,:] - mobject.points[i+1,:])
        if diff> normalDiff * 1.2:
            noteableThings +=[Line(mobject.points[i+2,:], mobject.points[i+1,:],color='#00ff00')]
            numbers = [Tex(i+2).move_to(mobject.points[i+2,:]).scale(0.4).set_fill(color='#0000ff').set_z_index(3),
            Tex(i+1).move_to(mobject.points[i+1,:]).set_fill(color='#0000ff').scale(0.4).set_z_index(3)]
            noteableThings += numbers
    #if len(noteableThings)>1:
    #    noteableThings = 
    return VGroup(*noteableThings)

def testLayer(func):
    def newFunc(t):
        print("testing function got used")
        return func(t)
    return newFunc

###########################################################################################################
############################################## Niche Tools ################################################
###########################################################################################################

class shortenName:
    def __init__(self,var) -> None:
        self.var = var
    
    def __enter__(self):
        return self.var
    
    def __getattr__(self,__name):
        if hasattr(getattr(self.animationType,__name),"__call__"):
            if not (__name in ["__enter__","__exit__"]):
                return partial(getattr(self.animationType,__name),self)
            else:
                return self.interpolate_submobject
        else:
            try:
                temp = getattr(self.animationType,__name)
            except:
                raise AttributeError("This OpacOverideAnimation has no attribute '"+__name+"'")
            self.prevUsedNonFuncs.append(__name)
            setattr(self,__name,temp)
            return temp


#For making pseudo-random rate functions that speed up and slow down strangely
def subdivide(prob,length,rang,seed: Seed,func = lambda x,y: linear(y),seedChange=3):
            if seed.rand(seedChange) < prob:
                x= rang[0] + seed.rand(seedChange) * (rang[1]-rang[0])
                y = seed.rand(seedChange)
                y=func(x,y)
                result =    list((*subdivide(
                                prob * (1-np.exp(-length * (rang[1]-rang[0]))),
                                length,
                                (rang[0],x),
                                seed,
                                func=func
                            ),)) + [(x,y)] + \
                            list((*subdivide(
                                prob * (1-np.exp(-length * (rang[1]-rang[0]))),
                                length,
                                (x,rang[1]),
                                seed,
                                func=func),))
                if prob==1:
                    return result + [(1., seed.rand(seedChange))]
                else:
                    return result
            else:
                return []

#I aint even gonna pretend I can explain this quickly
# https://www.overleaf.com/read/mmxxjmyybrmm
def turnFlatToFinitePDF(rang,PDF=None,CDF=None,samples=100):
    if PDF==None and CDF==None:
        raise ValueError("Must define one of PDF or CDF")
    if CDF!=None:
        x0 = np.linspace(rang[0],rang[1],num=samples)
        y0 = [CDF(t) for t in x0]
        x1 = np.linspace(0,1,num=samples)
        return (lambda func,a,b: lambda t: (b-a)*func(t) + a)(interp1d(y0,x1),rang[0],rang[1])
    else:
        x0 = np.linspace(rang[0],rang[1],num=samples)
        y0 = [PDF(x0[0])]
        for i in range(1,samples):
            y0 += [PDF(x0[i])/samples + y0[i-1]]
        y0=np.array(y0)
        y0 /= y0[-1]
        x1 = np.linspace(0,1,num=samples)
        return (lambda func,a,b: lambda t: (b-a)*func(t) + a)(interp1d(y0,x1),rang[0],rang[1])

def getBoundingAtAngle(object: Mobject,angle:float) -> np.ndarray:
    """
    Get bounding box list [center,*(each point
    around circle in pi/4 angle incriments)]
    """
    argList = [[0,0,0],RIGHT,RIGHT+UP,UP,UP+LEFT,LEFT,LEFT+DOWN,DOWN,DOWN+RIGHT]
    object.rotate(angle)
    result = np.array((*map(lambda t: object.get_critical_point(t),argList),))
    object.rotate(-angle)
    return result

###########################################################################################################
############################################## Aggregation ################################################
###########################################################################################################

#Generic
class VariableCollection(dict):
    def __init__(self,vars: Dict[str,Any]):
        super().__init__(vars)
        self.keyList = [key for key in vars]

    def __getattr__(self, __name: str) -> Any:
        #if hasattr(self,__name):
        #    return super().__getattr__(__name)
        try:
            super().__getattr__(__name)
        except:
            return self[__name]

    def __getitem__(self,key):
        if type(key) == int:
            return super().__getitem__(self.keyList[key])
        elif type(key) == slice:
            return super().__getitem__(
                [super().__getitem__(keys) for keys in super().__getattr__("keyList")[key]]
            )
        elif type(key) == str:
            return super().__getitem__(key)
        else:
            raise ValueError("Require int or str or slice")

    def __iter__(self):
        self.iterCounter = 0
        self.iterCountTo = len(self)
        return self

    def __next__(self):
        if self.iterCounter < self.iterCountTo:
            result = [self.keyList[self.iterCounter], self[self.iterCounter]]
            self.iterCounter += 1
            return result
        else:
            raise StopIteration

#VariableCollectionType = _GenericAlias()

class MobjectCollection(VGroup):
    def __init__(self,vars: Dict[str,Mobject]):
        super().__init__(*[items for key,items in vars.items()])
        self.innerDict = vars
        self.notMyProblem = dir(self)

    def __getattr__(self, __name: str) -> Any:
        if __name in super().__getattr__("innerDict"):
            return super().__getattr__("innerDict")[__name]
        else:
            super().__getattr__(__name)

    def __getitem__(self,key):
        if type(key) == int or type(key) == slice:
            return super().__getitem__(super().__getitem__(key))
        elif type(key) == str:
            return super().__getattr__("innerDict")[key]
        else:
            raise ValueError("Require int or str or slice")

class SequenceLayer:
    def __init__(self,
        mobjects: MobjectCollection | Dict[str,Mobject],
        animationsWithTimes: VariableCollection | Dict[str,List[Animation,float,float]],
        info: dict = None
    ):
        if type(mobjects) == dict:
            self.mobjects = MobjectCollection(mobjects)
        elif type(mobjects) == MobjectCollection:
            self.mobjects = mobjects
        else:
            raise ValueError(
            "Need MobjectCollection or Dictionary containing string-keyed mobjects"
            )
        if type(animationsWithTimes) == dict:
            self.animations = VariableCollection(animationsWithTimes)
        elif type(animationsWithTimes) == VariableCollection:
            self.animations = animationsWithTimes
        else:
            raise ValueError(
                """
                Need VariableCollection or Dictionary containing string-keyed lists of structure 
                [Animation, start_time, end_time]
                """[1:-1].replace("\t",""))

        self.info = info

class ElementaryAnimation:
    def __init__(self, animation: Animation, 
                startTime: float, zLayer: float,
                sustain: bool | str = False,
                ID: NoneType | str = None,
                info: Dict = dict() ):
        """ Stuff to go to the animation compiler.
        Animation is the animation to be played, starTime is the time it should start,
        zLayer is for what it should be rendered on top of. Sustain is set to false if 
        the animation's Mobjects should disappear after last frame however if it should
        stick around, set not to true but to an ID string that will incidicate when it
        should disappear. Info is a dictionary of other relevant information that isn't
        recognized for all elementary animations by default.
        """
        if not isinstance(animation,Animation):
            raise ValueError("Animation argument must be animation, else use CompoundAnimation")
        self.animation = animation
        self.startTime = startTime
        self.zLayer = zLayer
        self.sustain = sustain
        self.ID = ID
        self.info = info
    def __getitem__(self,key):
        if type(key) == int: return [self.animation, self.startTime, self.zLayer,
            self.sustain, self.ID, self.info][key]
        elif type(key) == str: 
            return getattr(self,key)

class CompoundAnimation(ElementaryAnimation):
    def __init__(self, *animations: VariableCollection | SequenceLayer, 
                startTime: float, zLayer: float,
                sustain: bool | str = False,
                ID: NoneType | str = None,
                info: Dict = dict() ):
        """ Stuff to go to the animation compiler.
        Animation is the animation to be played, starTime is the time it should start,
        zLayer is for what it should be rendered on top of. Sustain is set to false if 
        the animation's Mobjects should disappear after last frame however if it should
        stick around, set not to true but to an ID string that will incidicate when it
        should disappear. Info is a dictionary of other relevant information that isn't
        recognized for all elementary animations by default.
        """

        for i in animations:
            if not type(i) in [ElementaryAnimation, CompoundAnimation]:
                raise 
        self.animations = list(animations)
        self.startTime = startTime
        self.zLayer = zLayer
        self.sustain = sustain
        self.ID = ID
        self.info = self.info

    def addAnimations(self, *animations):
        for i in animations:
            if not type(i) in [ElementaryAnimation, CompoundAnimation]:
                raise 
        self.animations += list(animations)
    

class ElementaryImgSequence:
    def __init__(self,
    address: str, frameToStartOn: int,
    zlayer: float, duration: int,
    sustain: NoneType | str,
    ID: NoneType | str,
    info: dict = dict()):
        self.address = address 
        self.frameToStartOn = frameToStartOn
        self.zLayer = zlayer
        self.duration = duration
        self.sustain = sustain
        self.ID = ID
        self.info = info
        self.counterCheck: int = 0

    def shouldUseHere(self, currentFrame: int, lastBestZlayer: float):
        return \
            currentFrame >= self.frameToStartOn \
                and \
            currentFrame - self.frameToStartOn < self.duration
    
    #def getFrame(self, frameToGet: int, rendering: bool = True):






class CompoundImgSequence:
    def __init__(self,
    *initialContents: ElementaryImgSequence,
    frameToStartOn: int,
    zlayer: float, sustain: NoneType | str,
    ID: NoneType | str,
    info: dict = dict()):
        for i in initialContents:
            if type(i) != ElementaryImgSequence:
                raise ValueError
        self.frameToStartOn = frameToStartOn
        self.zLayer = zlayer
        self.sustain = sustain
        self.ID = ID
        self.info = info
        self.IDList = dict()
