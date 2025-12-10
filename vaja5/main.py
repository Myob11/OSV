import numpy as np
from OSV_lib import load_image, display_image
from OSV_lib import computeHistogram, displayHistogram, eqializeHistogram
from OSV_lib import InterpolateImage
import matplotlib.pyplot as plt

#%%
if __name__ == "__main__":

    I = load_image(r"vaja5\data\image-512x512-16bit.raw", [512,512], np.int16) 
    display_image(I, "Originalna slika")
    print(f"I: min = {I.min()}, max = {I.max()}")



def scaleImage(iImage, iK, iN): # če y = k*x + n imamo iK in iN naši parametri za scaling
    
    oImage = np.array(iImage, dtype=float)

    oImage = iImage * iK + iN
    
    return oImage

if __name__ == "__main__":
    scaledI = scaleImage(I, -0.125, 256)
    display_image(scaledI, "slika po lin preslikavi")
    print(f"I: min = {scaledI.min()}, max = {scaledI.max()}")

def WindowImage(iImage, iC, iW):

    oImage = np.array(iImage, dtype=float)
    oImage = 255/iW * (iImage - (iC - iW / 2))
    oImage[iImage < iC - iW / 2] = 0 # kar je pred oknom na 0
    oImage[iImage > iC + iW / 2] = 255 # kar je za oknom na 255

    return oImage


if __name__ == "__main__":
    scaledI = scaleImage(I, -0.125, 256)
    windowedI = WindowImage(scaledI, 1000, 500)
    display_image(windowedI, "slika po oknjenju")
    print(f"I: min = {scaledI.min()}, max = {scaledI.max()}")



def sectionalScaleImage(iImage , iS , oS):  # iS vhodno okno oS izhodno okno
    #prvi stolpec kje do kje , drugi kam do kam
    oImage = np.array(iImage, dtype=float)

    for i in range(len(iS) - 1):
        # najdemo spodnjo in zgornjo mejo intervala
        sLow = iS[i]
        sHigh = iS[i + 1]

        # idx je maska nase vhodne slike, z elementi ki so
        # v definiranem intervalu sLow sHigh
        idx = np.logical_and(iImage >= sLow, iImage <= sHigh) 

        # k  je razmerje med vhodno in izhodno skalo okna
        k = (oS[i+1] - oS[i]) / (sHigh - sLow)

        # realevantne dele (idx) izhodne slike linearno skaliram
        oImage[idx] = k * (iImage[idx] - sLow) + oS[i]

    return oImage


if __name__ == "__main__":
   sCP = np.array([[0, 85], #prvi stolpec kje do kje , drugi kam do kam
                    [85, 0],
                    [170, 255],
                    [255, 170]])
   sectionalScaledI = sectionalScaleImage(windowedI, sCP[:, 0], sCP[:,1])
   display_image(sectionalScaledI, "sectonal scaled image")
   print(sCP[:, 0], "prvi stolpec")
   print(sCP[:,1], "drugi stolpec")


def gammaImage(iImage , iG):

    oimage = np.array(iImage, dtype=float)

    oImage = 255 ** (1 - iG) * (iImage ** iG) # g = (L −1)^(1−γ) * f^γ

    return oImage


if __name__ == "__main__":

    gammaI = gammaImage(windowedI, 0.1)
    display_image(gammaI, "gamma preslikana slika")


