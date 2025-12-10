import numpy as np
from OSV_lib import load_image, display_image
from OSV_lib import computeHistogram, displayHistogram, eqializeHistogram
from OSV_lib import InterpolateImage
import matplotlib.pyplot as plt

#%%
def loadImage3D(iPath, iSize, iType):

    fid = open(iPath, "rb")
    # zamenjamo vrstni red x in y osi za prikaz slike
    im_shape = (iSize[1], iSize[0], iSize[2])
    oImage = np.ndarray(shape=im_shape, dtype=iType, buffer=fid.read(), order="F") #order ostane zaradi fortrana

    fid.close()
    return oImage

if __name__ == "__main__": #če ta file importamo drugam se ta del kode ne bo izvajal pri importih
    imSize = [512, 58, 907]
    pxDim = [0.597656, 3, 0.597656]
    I = loadImage3D(r"vaja4\data\spine-512x058x907-08bit.raw", imSize, np.uint8) #uint pomeni da so 8 bit vrednosti in so intigerji, niso negativne 
    print(I.shape)

    display_image(I[30, :, :], "Prerez slike")
    display_image(I[:,250 , :], "Prerez slike2")
    display_image(I[:, :, 450], "Prerez slike3")

def getPlanarCrossSection ( iImage , iDim , iNormVec , iLoc ):
    # inicializacija izhodnih argumentov
    Y, X, Z = iImage.shape
    dx, dy ,dz = iDim

    if iNormVec == [1, 0, 0]: #stranski prerez
        oCS = iImage[:, iLoc, :]
        #vektor slikovnih el. skaliramo v Y smer z Y velikostjo
        oV = np.arange(Y) * dy # .arrange nam da vektor do neke velikosti (npr od 0 do 521) krat velikosten razred pixla v to smer, dobimo vektor ki ima voxlov kot og slika,
                                #pixli niso vec veliki 1 ampak so skalirani z dejansko velikostjo pixla
        oH = np.arange(Z) * dz

    elif iNormVec == [0, 1, 0]:
        oCS = iImage[iLoc, :, :]
        oV = np.arange(X) * dx
        oH = np.arange(Z) * dz

    elif iNormVec == [0, 0, 1]:
        oCS = iImage[:, :, iLoc]
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    else:
        raise NotImplementedError("unknown normvec")

    return np.array(oCS) , oH , oV

if __name__ == "__main__":

    xc = 290
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [1, 0, 0], xc)
    display_image(sagCS, "segitalni prerez slike", sagH, sagV)

    xc = 30
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 1, 0], xc)
    display_image(sagCS, "segitalni prerez slike", sagH, sagV)

    xc = 450
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 0, 1], xc)
    display_image(sagCS, "segitalni prerez slike", sagH, sagV)

## planarna projekcija
def getPlanarProjection ( iImage , iDim , iNormVec , iFunc ):

    Y, X, Z = iImage.shape
    dx, dy ,dz = iDim   

    if iNormVec == [1, 0, 0]: # stranski prerez 
        oP = iFunc(iImage, axis = 1).T # transponiranje matrike , jo obrnemo
        oV = np.arange(Z) * dz 
        oH = np.arange(Y) * dy

    elif iNormVec == [0, 1, 0]:
        oP = iFunc(iImage, axis = 0).T # transponiranje matrike , jo obrnemo
        oV = np.arange(Z) * dz 
        oH = np.arange(X) * dx

    elif iNormVec == [0, 0, 1]:
        oP = iFunc(iImage, axis = 2) # transponiranje matrike , jo obrnemo
        oV = np.arange(Y) * dy
        oH = np.arange(X) * dx

    elif iNormVec[2] == 0:
        print("izpolnjen pogoj")

    else:
        raise NotImplementedError("unknown normvec")


    return oP, oH, oV


if __name__ == "__main__":
    func = np.max

    sagP, sagH, sagV = getPlanarProjection(I, pxDim, [1, 0, 0], func)
    display_image(sagP, "maximalna projekcija", sagH, sagV)

    sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 1, 0], func)
    display_image(sagP, "maximalna projekcija", sagH, sagV)

    sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 0, 1], func)
    display_image(sagP, "maximalna projekcija", sagH, sagV)

# %%

## dodatno gradivo

# 1

print("DODATNO GRADIVO")

xc = 256
stranskiCS, stranskiH, stranskiV = getPlanarCrossSection(I, pxDim, [1,0,0], xc)
display_image(stranskiCS, "stranski prerez na položaju xc = 256 slikovnih elementov", stranskiH, stranskiV)

yc = 35
celniCS, celniH, celniV = getPlanarCrossSection(I, pxDim, [0,1,0], yc)
display_image(celniCS, "čelni prerez na položaju yc = 35 slikovnih elementov", celniH, celniV)

zc = 467
precniCS, precniH, precniV = getPlanarCrossSection(I, pxDim, [0,0,1], zc)
display_image(precniCS, "prečni prerez na položaju zc = 467 slikovnih elementov", precniH, precniV)

# 2
# max vrednost
func = np.max

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [1, 0, 0], func)
display_image(sagP, "maximalna projekcija", sagH, sagV)

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 1, 0], func)
display_image(sagP, "maximalna projekcija", sagH, sagV)

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 0, 1], func)
display_image(sagP, "maximalna projekcija", sagH, sagV)

# povprečna vrednost

func = np.average

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [1, 0, 0], func)
display_image(sagP, "povprečna projekcija", sagH, sagV)

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 1, 0], func)
display_image(sagP, "povprečna projekcija", sagH, sagV)

sagP, sagH, sagV = getPlanarProjection(I, pxDim, [0, 0, 1], func)
display_image(sagP, "povprečna projekcija", sagH, sagV)


