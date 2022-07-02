from __future__ import annotations
from ctypes.wintypes import tagPOINT
from typing import Callable, List, Any
from manim import *
import itertools as it
from multiprocess import Process, Pool
import numpy as np

MAXTHREADS = 8

def binaryMerge(mergeOverThis: List[Any],mergeOperation: Callable, maxNumOfThreads = MAXTHREADS):
    while len(mergeOverThis) > 1:
        length = len(mergeOverThis)
        halfishLength = int(np.floor(length / 2))
        processes = np.zeros(halfishLength,dtype=object)
        inOut = []
        isEven = length % 2 == 0
        numOfThreads = halfishLength if halfishLength < maxNumOfThreads else maxNumOfThreads
        if isEven:
            for i in range(halfishLength):
                inOut += [mergeOverThis[i*2:i*2+2]]
            with Pool(numOfThreads) as p:
                inOut = list(p.map(mergeOperation, inOut))
        elif not isEven:
            for i in range(halfishLength):
                inOut += [mergeOverThis[i*2:i*2+2]]
            with Pool(numOfThreads) as p:
                inOut = list(p.map(mergeOperation, inOut)) + [mergeOverThis[-1]]
        mergeOverThis = inOut
    return mergeOverThis
            
def imgMerge(img1, img2):
    """Must have same dimensions and alpha channel, img2 on top of img1"""
    return img1[:,:,0:3] * img1[:,:,3] * (1 - img2[:,:,3]) + img2[:,:,0:3] * img2[:,:,3]


class myCamera(Camera):
    def display_multiple_non_background_colored_vmobjects(self, vmobjects, pixel_array):
        vmobjects = list(vmobjects)
        numOfvmobjects = len(vmobjects)
        pixelArrays = [np.zeros(pixel_array.shape,dtype=self.pixel_array_dtype) for i in range(numOfvmobjects)]
        ctxArrays = [self.get_cairo_context(pixelArray) for pixelArray in pixelArrays]
        #processes = []
        #for vmobject,ctx in zip(vmobjects,ctxArrays):
        #    processes += [Process(target = self.display_vectorized, args = (vmobject, ctx))]
        #    processes[-1].start()
        #for i in processes:
        #    i.join()
        args = [(self,*i) for i in zip(vmobjects,ctxArrays)]
        with Pool(MAXTHREADS) as p:
            p.map(self.display_vectorized,args)
        result = binaryMerge(pixelArrays, imgMerge)
        self.pixel_array = imgMerge(self.pixel_array, result)
        

