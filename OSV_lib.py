from matplotlib import pyplot as plt
import numpy as np 


### 1.lab
def display_image(iImage, iTitle="", iGridX = None, iGridY = None, cmap = 'gray'):
    plt.figure()
    plt.title(iTitle)

    if iGridX is None and iGridY is None:
        extent = (
            -0.5,
            iImage.shape[1] - 0.5,
            iImage.shape[0] - 0.5,
            -0.5
        )

    else:
        stepX = iGridX[1] - iGridX[0]
        stepY = iGridY[1] - iGridY[0]

        extent = (
            iGridX[0] - stepX/2,
            iGridX[-1] + stepX/2,
            iGridY[-1] + stepY/2,
            iGridX[0] - stepY/2,
        )

    plt.imshow(
        iImage,
        cmap=cmap,
        aspect="equal",
        extent=extent,
    )
    
    plt.show()


def load_image(path, size, type):
    #odpremo datoteko
    fid = open(path, "rb")
    #preberemo vsebino
    buffer = fid.read()
    # dobimo dolzino prebrane vsebine
    buffer_len = len(np.frombuffer(buffer, type))
    # dobi stevilo elementov v sliki
    stevilo_elementov = np.prod(size) #npr 500x400
    if stevilo_elementov == buffer_len:
        oImageShape = (size[1], size[0]) #visina, sirina
    else:
        oImageShape = (size[1], size[0], 3) #visina, sirina
    #ustvari izhodni array v 2d obliki
    oImage = np.ndarray(oImageShape, type, buffer, order="F")
    # zapri datoteko
    fid.close()
    return oImage

## dodatno gradivo
def saveImage(iImage, iPath, iType):

    fid = open(iPath, "wb")
    oImage = np.array(iImage, dtype=iType)
    # zapisemo v Fortranskem nacinu, da lahko za nalaganje te datoteke uporabimo naso loadImage funkcijo
    fid.write(oImage.tobytes(order="F"))

    fid.close()
    return

# ********************************************************************************************************************************** #

### 2.lab

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

def addNoise ( iImage , iStd ):
    oNoise = np.random.randn(*iImage.shape) * iStd   # float noise, lahko negativna
    oImage_f = iImage.astype(np.float64) + oNoise
    oImage_f = np.clip(oImage_f, 0, 255)
    oImage = oImage_f.astype(np.uint8)
    return oImage, oNoise

# ********************************************************************************************************************************** #

### 3.lab

def InterpolateImage(iImage, iSize, iOrder):

    iOrder = int(iOrder)
    # velikost vhodne slike
    Y,X = iImage.shape
    # veliksot izhodne slike
    M,N = iSize
    # inicializacije izhodne slike
    oImage = np.zeros((N, M), dtype=iImage.dtype) #naredimo matriko z 0, velikosti N,M, naredimo isto type kot iImage
    
    dx = (X-1) / (M-1)
    dy = (Y-1) / (N-1)

    for n in range(N): # vrstice izhodne slike
        for m in range(M): # stolpci izhodne slike
            # točka v vhodnem koordinatnem sistemu
            pt = np.array([m*dx, n*dy]) # pri (1, 1) na izhodni sliki smo na (1/3, 1/2) vhodne slike!

            if iOrder == 0: # vzamemo vrednost najbližjega soseda
                px = np.round(pt).astype(int) # najblizji pixel nasi poziciji v arrayu, želimo da je tip intiger
                s = iImage[px[1],px[0]]  # px se spreminja po tabeli [1] nam pove da se premikamo po vrsticah, [0] da se po stolpcih
            
            elif iOrder == 1: # računamo ploščine !
                px = np.floor(pt).astype(int) # želimo se premakniti v eno točko
                a = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 1)) # da ne pride do negativnih ploščin uporabimo abs
                b = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 1))
                c = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 0))
                d = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 0))

                sa = iImage[px[1], px[0]] # sivinska vrednost ki pripada ploščini a
                sb = iImage[px[1], min(px[0] + 1, X-1)] 
                sc = iImage[min(px[1] + 1, Y-1), px[0]]
                sd = iImage[min(px[1] + 1, Y-1), min(px[0] + 1, X-1)]

                s = int(sa*a + sb*b + sc*c + sd*d)

            else:
                raise NotImplementedError # če ni nič od naštetega vrži ta error
            oImage[n, m] = s # na izhodni sliki zapišemo na to mesto vrednost s!

    return oImage

## dodatno gradivo

def decimateImage(iImage , iKernel , iLevel = 1):

    Y , X = iKernel.shape
    n = (Y - 1) // 2
    m = (X - 1) // 2

    novaSlika = np.array(iImage)

    # pad image by y rows and x cols (use reflect to avoid border artifacts)
    novaSlika = np.pad(novaSlika, ((n, n), (m, m)), mode='reflect')

    iY, iX = novaSlika.shape

    oImage = np.zeros(novaSlika.shape, dtype=float)

    # del slike enako velik kot iKernel, ki ga bova pomnzila z iKernel in 
    # nato seštela vrednost v pixel na lokaciji (x,y)

    for y in range(n, iY - n):
        for x in range(m, iX - m):
            patch = novaSlika[y - n : y + n + 1, x - m: x + m + 1]
            oImage[y,x] = (patch * iKernel).sum()

    oImage = oImage[n : iY - n, m : iX - m]

    for level in range(iLevel):
        oImage = oImage[::2,::2]
    
    print("oimage shape je: ", oImage.shape)

    return oImage

# ********************************************************************************************************************************** #

### 4.lab

def loadImage3D(iPath, iSize, iType):

    fid = open(iPath, "rb")
    # zamenjamo vrstni red x in y osi za prikaz slike
    im_shape = (iSize[1], iSize[0], iSize[2])
    oImage = np.ndarray(shape=im_shape, dtype=iType, buffer=fid.read(), order="F") #order ostane zaradi fortrana

    fid.close()
    return oImage

def getPlanarProjection ( iImage , iDim , iNormVec , iFunc ) :

    Y, X, Z = iImage.shape
    dx, dy ,dz = iDim    

    if iNormVec == [1, 0, 0]: #stranski prerez 
        oP = iFunc(iImage, axis = 1).T #transponiranje matrike , jo obrnemo
        oV = np.arange(Z) * dz 
        oH = np.arange(Y) * dy

    elif iNormVec == [0, 1, 0]:
        oP = iFunc(iImage, axis = 0).T #transponiranje matrike , jo obrnemo
        oV = np.arange(Z) * dz 
        oH = np.arange(X) * dx

    elif iNormVec == [0, 0, 1]:
        oP = iFunc(iImage, axis = 0) #transponiranje matrike , jo obrnemo
        oV = np.arange(Y) * dy
        oH = np.arange(X) * dx

    else:
        raise NotImplementedError("unknown normvec")


    return oP, oH, oV

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

## dodatno gradivo



# ********************************************************************************************************************************** #

### 5.lab

def scaleImage(iImage, iK, iN): # če y = k*x + n imamo iK in iN naši parametri za scaling
    
    oImage = np.array(iImage, dtype=float)
    oImage = iImage * iK + iN
    
    return oImage

def WindowImage(iImage, iC, iW):

    oImage = np.array(iImage, dtype=float)
    oImage = 255/iW * (iImage - (iC - iW / 2))
    oImage[iImage < iC - iW / 2] = 0 # kar je pred oknom na 0
    oImage[iImage > iC + iW / 2] = 255 # kar je za oknom na 255

    return oImage


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


def gammaImage(iImage , iG):

    oimage = np.array(iImage, dtype=float)
    oImage = 255 ** (1 - iG) * (iImage ** iG) # g = (L −1)^(1−γ) * f^γ

    return oImage

## dodatno gradivo

# ********************************************************************************************************************************** #

### 6.lab

def getRadialValue(iXY, iCP):
    # k stevilo kontrolnih točk
    k = iCP.shape[0]

    oValue = np.zeros(k)

    x_i, y_i = iXY
    for i in range(k): # for loop ki gre cez vse kontrolne točke
        x_k, y_k = iCP[i]

        # razdalja med točko in kontrolno točko
        radius = np.sqrt((x_i - x_k) ** 2 + (y_i - y_k) ** 2)

        # vrednost radialne funkcije po formuli
        if radius > 0:
            oValue[i] = -(radius ** 2) * np.log(radius)

    return oValue

def getParameters(iType, scale = None, trans = None, rot = None, shear = None, original_points = None, mapped_points = None):
    # lahko bi ze v paramtreih za nedefirniane spremenljivke nastavili vrednosti
    # npr scale = [1, 1]
    oP = {}

    # inicializiramo matrike, ki ne predstavljajo spremembe
    if iType == "affine":
        if scale is None:
            scale = [1, 1]
        if trans is None:
            trans = [0, 0]
        if rot is None:
            rot = 0
        if shear is None:
            shear = [0, 0]
        
        # matrika skaliranja
        Tscale = np.array([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [0,    0,     1]
        ]) 

        Ttrans = np.array([
            [1, 0, trans[0]],
            [0, 1, trans[1]],
            [0,    0,     1] 
        ])

        phi = rot * np.pi / 180
        Trot = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi),  0],
            [0,            0,           1]
        ])

        Tshear = np.array([
            [1,  shear[0], 0],
            [shear[1], 1, 0],
            [0,    0,     1]
        ])

        # @ vektorsko množenje matrik!!! (matrično množenje)
        oP = Tshear @ Trot @ trans @ Tscale

        return oP

def transformImage( iType , iImage , iDim , iP , iBackground = 0 , iInterp = 0):

    Y, X = iImage.shape
    oImage  = np.ones((Y,X)) * iBackground
    dx, dy = iDim

    for y in range(Y):
        for x in range(X):
            # indeks slikovnega elementa --> trenutna točka
            x_hat, y_hat = x * dx, y * dy # x_hat je x s stresico iz matematike

            if iType == "affine":
                # zmnozimo pixel z affino matriko
                x_hat, y_hat, _ = iP @ np.array([x_hat, y_hat, 1])
            
            elif iType == "radial":
                # radialno preslikamo tocke glede na koeficiente radialne preslikave
                U = getRadialValue([x_hat, y_hat], iP["pts"])
                x_hat, y_hat = np.array([U @ iP["alphas"], U @ iP["betas"]])
            
            # preslikamo nazaj v indeks  slikovnega elementa
            x_hat = x_hat/dx
            y_hat = y_hat/dy

            if iInterp == 0:
                # zaokrozimo vrednosti na cele vrednosti slikovnih indeksov
                x_hat, y_hat = round(x_hat), round(y_hat)
                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    # prepisemo vrednost iz vhodne slike na izhodno na novi lokaciji slikovnega elementa
                    oImage[y, x] = iImage[y_hat, x_hat]
    
    """      elif iInterp == 1:
                
                dx = (X-1) / (M-1)
                dy = (Y-1) / (N-1)

                px = np.floor(pt).astype(int) # želimo se premakniti v eno točko
                a = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 1)) # da ne pride do negativnih ploščin uporabimo abs
                b = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 1))
                c = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 0))
                d = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 0))

                sa = iImage[px[1], px[0]] # sivinska vrednost ki pripada ploščini a
                sb = iImage[px[1], min(px[0] + 1, X-1)] 
                sc = iImage[min(px[1] + 1, Y-1), px[0]]
                sd = iImage[min(px[1] + 1, Y-1), min(px[0] + 1, X-1)]

                s = int(sa*a + sb*b + sc*c + sd*d)
                """

    return oImage

## dodatno gradivo

# ********************************************************************************************************************************** #

### 7.lab

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

## dodatno gradivo

# ********************************************************************************************************************************** #
