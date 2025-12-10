
#%%
import numpy as np
import matplotlib.pyplot as plt
import os, sys

parent_folder = os.getcwd()
sys.path.append(parent_folder)

from OSV_lib import load_image, display_image




## 2. NALOGA

# funkcija za izračun histograma slike
def computeHistogram(iImage):

    #poiščemo pt. bitov za zapis obmocja od 0 do Lmax
    nBits = int(np.log2(iImage.max())) + 1  #preslikava v int samo odreže decimalno vrednost

    # dolocimo vektor sivinskih vrednosti opazovanega obmocja
    oLevels = np.arange(0,2 ** nBits, 1) #definira vektor od kje do kje , z kaksnim korakom

    #
    oHist = np.zeros(len(oLevels)) #rezerviramo plac za celoten vektor! Resimo se mem. problemov

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y,x]] = oHist[iImage[y,x]] + 1

    # delimo z vsoto pixlov da dobimo normaliziran historgram
    oProb = oHist / iImage.size

    oCDF = np.zeros_like(oHist)

     #izracun CDF

    for i in range(len(oProb)):
        oCDF[i] = oProb[: i + 1].sum() # : i + 1 pomeni da gremo cez cel array do i in sestejemo
    
    return oHist, oProb, oCDF, oLevels

#funkcija za izris histograma
def displayHistogram(iHist, iLevels, iTitle):

    plt.figure()
    plt.title(iTitle)
    plt.bar(iLevels, iHist, width=1)
    plt.xlim((iLevels.min(), iLevels.max()))
    plt.ylim((0,1.05*iHist.max()))
    plt.show()
    return

def eqializeHistogram(iImage):
    _, _, cdf, _ = computeHistogram(iImage)

    nBits = int(np.log2(iImage.max())) + 1

    max_intensity = 2 ** nBits - 1

    oImage = np.zeros_like(iImage)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y,x]
            new_intensity = np.floor(cdf[old_intensity] * max_intensity)
            oImage[y,x] = new_intensity

    return oImage

if __name__ == "__main__":
   ## 1. NALOGA
   I = load_image("vaja2/data/valley-1024x683-08bit.raw", (1024, 683), np.uint8)
   display_image(I, "Originalna slika")

   ## 2. NALOGA

   h, p, c, l = computeHistogram(I)


   displayHistogram(h, l, "histogram")
   displayHistogram(p, l, "normalizirani histogram")
   displayHistogram(c, l, "CDF")

   I_eq = eqializeHistogram(I)

   display_image(I_eq, "ekvalizirana slika")

   h_eq, p_eq, c_eq, l_eq = computeHistogram(I_eq)
   displayHistogram(h_eq, l_eq, "ekvaliziran histogram")
   displayHistogram(p_eq, l_eq, "ekvaliziran normaliziran histogram")
   displayHistogram(c_eq, l_eq, "ekvaliziran normaliziran histogram")



## dodatno gradivo

def computeEntropy(iImage):
    _, p, _, _ = computeHistogram(iImage)
    oEntropy = 0
    for prob in p:
        if prob == 0:
            continue
        else:
            oEntropy = oEntropy + (prob * np.log2(prob))
    oEntropy = -oEntropy
    return oEntropy

if __name__ == "__main__": 

    print("Entropija za sliko je: ", computeEntropy(I))
    print("Entropija za ekvalizirano sliko je: ", computeEntropy(I_eq))



def addNoise ( iImage , iStd ):
    oNoise = np.random.randn(*iImage.shape) * iStd   
    oImage_f = iImage.astype(np.float64) + oNoise
    oImage_f = np.clip(oImage_f, 0, 255)
    oImage = oImage_f.astype(np.uint8)
    return oImage, oNoise

if __name__ == "__main__":

    display_image(I, "Originalna slika")

    for std in [2, 5, 10, 25]:
        I_noise, noise = addNoise(I, std)
        display_image(I_noise, f"slika z dodanim šumom {std}")
        print("Noise matrika: ", noise)

# %%