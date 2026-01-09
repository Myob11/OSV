
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
frame2 = loadFrame(video, 51)
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
    PTS = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ])

    for n in range(N):
        # definiramo koordinate v y smer vsakega bloka
        y_min = n * dy
        y_max = (n + 1) * dy
        y = np.arange(y_min, y_max)

        for m in range(M):
            # enako za x smer
            x_min = m * dx
            x_max = (m + 1) * dx
            x = np.arange(x_min, x_max)

            # vsak center bloka shranimo v vektor
            oCenterPoints[n, m, 0] = x.mean()
            oCenterPoints[n, m, 1] = y.mean()

            # trenutni blok na sliki 2
            block2 = iFrame2[y_min:y_max, x_min:x_max]

            # logaritemsko iskanje vektorja premika
            for i in range(1, 4):
                # področje iskanja skalirano logaritemsko
                Pi = (P + 1) / (2 ** i)
                PTSi = PTS * Pi
                # definiramo kandidat prve opcije vektorja premika

                d0 = oMovingVector[n, m, :]

                for p in range(PTSi.shape[0]):
                    # vektor premika v smer PTS
                    d = d0 + PTSi[p, :]

                    # napoved bloka iz prve slike
                    predictedFrame2 = framePrediction(iFrame1, d)
                    # ujemajoči blok frame2 slike z premaknjeno frame1 sliko glede na koordinate bloka
                    predictedBlock2 = predictedFrame2[y_min:y_max, x_min:x_max]

                    # maska, kjer so od prej elementi ki smo jih nastavili na -1 ignorirani (0), vse ostalo pa je relevantno (1)
                    mask = predictedBlock2 >= 0

                    # absolutna napaka med drugo sliko in napovedano sliko iz prve slike
                    bErr = np.mean(np.abs(block2[mask] - predictedBlock2[mask]))

                    # če je napaka manjsa kot prejsnji boljsi rezultat si to zapomnimo
                    if bErr < Err[n, m]:
                        Err[n, m] = bErr
                        # za ta soecificn blok si tudi zapomnemo kateri premik je biu najboljsi
                        oMovingVector[n, m, :] = d
    
    return oMovingVector, oCenterPoints

if __name__ == "__main__":
    bSize = [8, 8]
    searchSize = 2 ** 4 - 1
    MV, CP = blockMatching(frame1, frame2, bSize, searchSize)
    print(CP)
    print(MV)

def displayMotionField(iMovingVector, iControlPoints, iTitle, iImage = None):
    if iImage is None:
        fig = plt.figure()
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title(iTitle)

    else:
        fig = display_image(iImage, iTitle)
    
    plt.quiver(
        iControlPoints[:, :, 0],
        iControlPoints[:, :, 1],
        iMovingVector[:, :, 0],
        iMovingVector[:, :, 1],
        color='r',
        angles='xy',
        units='xy',
        scale=0.5
    )

    return fig

if __name__ == "__main__":
    fig1 = displayMotionField(MV, CP, iTitle="Vektor premikov")
    fig2 = displayMotionField(MV, CP, iTitle="superponirani vektorji na og sliko", iImage=frame1)
