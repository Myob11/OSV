
from matplotlib import pyplot as plt
from OSV_lib import display_image
import numpy as np

import cv2

#%%

def loadFrame(iVideo, iK):
    # postavimo bralec videa na iK - ti frame
    iVideo.set(1, iK - 1)
    # preberemo frame
    ret, oFrame = iVideo.read()
    oFrame = oFrame[:, :, 0].astype(float)

    return oFrame

if __name__ == "__main__":
    video = cv2.VideoCapture("vaja11/data/simple-video.avi")
    print(f"stevilo frameov v videu: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")


frame1 = loadFrame(video, 50)
frame2 = loadFrame(video, 52)
display_image(frame1, "Frame 50")
display_image(frame2, "Frame 51")

def framePrediction(iFrame, iMovingVector):
    # preverimo da imamo cele stevilke
    iMovingVector = np.array(iMovingVector).astype(int)
    dx, dy = iMovingVector

    # premaknemo sliko v smer premika po x in y osi
    oFrame = np.roll(iFrame, [dy, dx], axis=(0, 1))

    # preliv vrednosti psotavimo na -1
    if dx >= 0:
        # premik v pozitivno smer, na .1 postavimo vse nove izmisljene piksle
        oFrame[:, :dx] = -1
        # če je dx negativen, se slika premakne v levo. Ker slika [ -N ] pomeni štetje od konca proti začetku
        # (v nasprotni smeri), -N: pomeni od pred N-tega elementa do konca
    else:
        oFrame[:, dx:] = -1

    # isto za y !!!
    if dy >= 0:
        oFrame[:dy, :] = -1
    else:
        oFrame[dy:, :] = -1

    return oFrame

if __name__ == "__main__":
    
    predictedFrame2 = framePrediction(frame1, [2, 4])
    # bitno obmocje je od 0 do 255
    substractedFrame2 = (predictedFrame2 - frame1 + 255) / 2
    display_image(substractedFrame2, "Predicted Frame 52")

# %%

def blockMatching(iFrame1, iFrame2, iBlockSize, iSearchSize):

    # velikost slike
    Y,X = iFrame1.shape
    dx, dy = iBlockSize

    # stevilo blokov v sliki
    M = int(X / dx)
    N = int(Y / dy)

    ## inicializiramo relevantne vektorje
    # vektor premikov za vsak blok
    oMovingVector = np.zeros((N, M, 2), dtype = int)
    # vektor centralnih točk za vsak blok
    oCenterPoints = np.zeros((N, M, 2), dtype = float)
    # vektor napak za vsak blok
    Err = np.ones((N, M), dtype = float) * 255

    # inicializiramo logaritemsko skalo vektorja premika
    # searchsize je definiran kot premer zato delimo z 2 da se lahko max premaknemo v 
    # katero koli smer za radij( / 2 search siza)
    P = (iSearchSize - 1) / 2
    PTS = np.array([])
