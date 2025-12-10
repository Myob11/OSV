from OSV_lib import load_image, display_image
import numpy as np

#%%
if __name__ == '__main__':
    I = load_image(r"vaja7\data\cameraman-256x256-08bit.raw", [256,256], np.uint8)
    display_image(I, "Originalna slika")

# funkcijo bomo uporabili znotraj spatial filtering zato mora biti definirana pred njo
def changeSpatialDomain ( iType , iImage , iX , iY , iMode = None , iBgr = 0) :

    Y, X = iImage.shape

    if iType == "enlarge":

        if iMode is None:
            # če ni načina defaultamo na 0
            # izhodna slika je vhodna slika + 2 * korak (iz filtra)
            oImage = np.zeros((Y + 2 * iY, X + 2 * iX))
            # v notranji del vstavimo vhodno sliko
            oImage[iY : Y + iY, iX : X + iX] = iImage

        elif iMode == "constant":
            # isto kot pri None samo da paddanje nastavimo na vrednost iBgr
            oImage = np.ones((Y + 2 * iY, X + 2 * iX)) * iBgr 
            # v notranji del vstavimo vhodno sliko
            oImage[iY : Y + iY, iX : X + iX] = iImage
        
        elif iMode == "extrapolation":
            oImage = np.array(iImage)
            # ponovimo, kolikor imamo veliko jedro(stevilo korakov)

            for y in range(iY):
                # zlepimo zgornjo vrstico slike, celotno sliko in nato se spodnjo vrstico slike
                oImage = np.vstack([oImage[0, :], oImage, oImage[-1, :]])

            for y in range(iX):
                # reshape ker je vektor vedno 1 dimenzionalen in je vedno v isti smeri
                # enako kot v x smer, vendar moramo vrstico za 90 stopinj zarotirat --> iz -- v | 
                # tako da lahko vrstico zlepimo k sliki
                oImage = np.hstack([oImage[:, 0].reshape(-1, 1), oImage, oImage[:, -1].reshape(-1, 1)])
        
        elif iMode == "reflection":
            oImage = np.array(iImage)

            for y in range(iY):
                # ker se premikamo po novih vrsticah se oddaljenost od zeljene vrednosti za prepis ustrezno vecja
                # ko dodajamo elemente se slika veča
                # posledicno moramo mnoziti korak * 2
                oImage = np.vstack(
                    [oImage[2 * y, :], 
                     oImage, 
                     oImage[-2 * y - 1, :]])

            for x in range(iX):
                oImage = np.hstack(
                    [oImage[:, 2 * x].reshape(-1, 1),
                     oImage,
                     oImage[:, -2 * x].reshape(-1, 1)])
                
        elif iMode == "period":
            oImage = np.array(iImage)
 
            oImage = np.array(iImage)

            for y in range(iY):
                # ker se premikamo po novih vrsticah se oddaljenost od zeljene vrednosti za prepis ustrezno vecja
                # ko dodajamo elemente se slika veča
                # posledicno moramo mnoziti korak * 2
                oImage = np.vstack(
                                [oImage[-2 * y - 1, :],
                                 oImage, 
                                 oImage[2 * y, :]])

            for x in range(iX):
                oImage = np.hstack(
                    [oImage[:, -2 * x - 1].reshape(-1, 1),
                     oImage,
                     oImage[:, 2 * x].reshape(-1, 1)])  

        else:
            NotImplementedError 

    elif iType == "reduce":
        oImage = iImage[iY : Y - iY, iX : X - iX]        

    return oImage

if __name__ == "__main__":

    xPad = 128
    yPad = 384

    spatialImage = changeSpatialDomain("enlarge", I, iX=xPad, iY=yPad, iMode="constant", iBgr=128)
    display_image(spatialImage, "razsirjena slika konst")

    spatialImage = changeSpatialDomain("enlarge", I, iX=xPad, iY=yPad, iMode="period", iBgr=128)
    display_image(spatialImage, "razsirjena slika period")

    spatialImage = changeSpatialDomain("enlarge", I, iX=xPad, iY=yPad, iMode="reflection", iBgr=128)
    display_image(spatialImage, "razsirjena slika reflection")

    spatialImage = changeSpatialDomain("enlarge", I, iX=xPad, iY=yPad, iMode="extrapolation", iBgr=128)
    display_image(spatialImage, "razsirjena slika extrapolation")

    spatialImage = changeSpatialDomain("enlarge", I, iX=xPad, iY=yPad, iBgr=128)
    display_image(spatialImage, "razsirjena slika None")
#%%

def spatialFiltering(iType, iImage, iFilter, iStatFunc=None, iMorphOp=None):
    
    N, M = iFilter.shape

    # premik v vse smeri glede na velikost jedra filtra (1 levo 1 desno 1 gor 1 dol)
    m = int((M - 1) / 2)
    n = int((N - 1) / 2)

    if iMorphOp == "erosion":
        # Pri eroziji bomo jemali min vrednost v jedru zato nastavimo background
        # na max da nam nebo skakal v zelje
        iBgr = 255
    else:   
        iBgr = 0

    # povecamo vhodno sliko za velikost jedra
    iImage = changeSpatialDomain("enlarge", iImage, m, n, iMode = "reflection", iBgr=iBgr)

    Y, X = iImage.shape

    oImage = np.zeros((Y,X), dtype=float)
    # premikamo se znotraj slike s tem, da upostevamo velikost jedra
    for y in range(n, Y - n):
        for x in range(m, X-m):
            # po patchih se premikamo po sliki 
            # 1:4 --> 1,2,3  NE 1,2,3,4 !!!
            # zato je y + n + 1 
            patch = iImage[y - n: y + n + 1,
                           x - m: x + m + 1]
            
            if iType == "kernel":
                # najprej zmnozimo elemente patcha s filtrom in hij nato sestejemo v vrednost
                oImage[y, x] = (patch * iFilter).sum()
            
            elif iType == "statistical":
                # izvedemo funkcijo na celem patchu
                # neko funkcijo izvedemo na patchu in nato elemte sestejemo v sredinsko vrednost
                oImage[y, x] = iStatFunc(patch)

            elif iType == "morphological":
                # upoštevamo samo piksle, kjer je jedro definirano oz. ni enako 0
                R = patch[iFilter != 0]
                
                if iMorphOp == "erosion":
                    oImage[y, x] = R.min()

                elif iMorphOp == "dilation":
                    oImage[y, x] = R.max()

                else:
                    print("unknown morphological operation", iMorphOp)
                    raise NotImplementedError

            else:
                print("unknown operation", iType)
                raise NotImplementedError
    
    oImage = changeSpatialDomain("reduce", oImage, m, n)
    return oImage

            
if __name__ == "__main__":
    
    K = np.array([[1,  1,  1],
                  [1, -8,  1],
                  [1,  1,  1]])
    
    KernelImage = spatialFiltering("kernel", I, iFilter = K)
    display_image(KernelImage, "slika filtrirana s kernelom")

    K = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])
    
    statisticalImage = spatialFiltering("statistical", I, iFilter = K, iStatFunc=np.average)
    display_image(statisticalImage, "slika filtrirana s statisticnim filtrom")
    
    Kmorph = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])

    morphologicalImage = spatialFiltering("morphological", I, iFilter = Kmorph, iMorphOp = "dilation")
    display_image(morphologicalImage, "morfolosko filtrirana slika dilacija")

    morphologicalImage = spatialFiltering("morphological", I, iFilter = Kmorph, iMorphOp = "erosion")
    display_image(morphologicalImage, "morfolosko filtrirana slika erozija")
    

