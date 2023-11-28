
import numpy as np
import math

def RotateHlp(matA, angDeg, a0, a1):
    matB = np.copy(matA)
    ang = math.radians(angDeg)
    sinAng, cosAng = math.sin(ang), math.cos(ang)
    for i in range(0, 4):
        matB[a0, i] = matA[a0, i] * cosAng + matA[a1, i] * sinAng
        matB[a1, i] = matA[a0, i] * -sinAng + matA[a1, i] * cosAng
    return matB

def RotateX(matA, angDeg):
    return RotateHlp(matA, angDeg, 1, 2)

def RotateY(matA, angDeg):
    return RotateHlp(matA, angDeg, 2, 0)

def RotateZ(matA, angDeg):
    return RotateHlp(matA, angDeg, 0, 1)

def RotateView(matA, angDeg):
    return RotateZ(RotateY(RotateX(matA, angDeg[0]), angDeg[1]), angDeg[2])

def Perspective(fov, aspectRatio, near, far):
    fn, f_n = far + near, far - near
    r, t = aspectRatio, 1.0 / math.tan(math.radians(fov) / 2.0)
    return np.matrix([[t / r, 0, 0, 0], [0, t, 0, 0], [0, 0, -fn / f_n, -2.0 * far * near / f_n], [0, 0, -1, 0]])

def Fract(val): return val - math.trunc(val)

def Translate(matA, trans):
    matB = np.copy(matA)
    for i in range(0, 4):
        matB[3, i] = matA[0, i] * trans[0] +\
                     matA[1, i] * trans[1] +\
                     matA[2, i] * trans[2] +\
                     matA[3, i]
    return matB


def Scale(matA, s):
    matB = np.copy(matA)
    for i0 in range(0, 3):
        for i1 in range(0, 4): matB[i0, i1] = matA[i0, i1] * s[i0]
    return matB