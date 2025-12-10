import numpy as np
from OSV_lib import load_image, display_image
from OSV_lib import computeHistogram, displayHistogram, eqializeHistogram
from OSV_lib import InterpolateImage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    imSize = [256, 512]
    pxDim = [2, 1]
    gX = np.arange(imSize[0]) * pxDim[0]
    gY = np.arange(imSize[1]) * pxDim[1]
    I = load_image(r"vaja6\data\lena-256x512-08bit.raw", imSize, np.uint8)
    display_image(I, "OG slika", gX, gY)

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
        oP = Tshear @ Trot @ Ttrans @ Tscale
    
    elif iType == "radial":
        assert original_points is not None
        assert mapped_points is not None 
        # preverjamo neko stanje ki zelimo da je res, če to ni res nas vrže iz funkcije!

        #stevilo kontrolnih točk
        K = original_points.shape[0]

        # inicializiranje matrike koeficientov
        UU = np.zeros((K, K), dtype=float)

        # zanka cez vse kontrolne točke
        for i in range(K):
            # vektor radialnih vrednosti med originalnimi kontrolnimi točkami
            UU[i, :] = getRadialValue(original_points[i, :], original_points)

        # alfa koeficienti med originalnimi in mapped tockami po x koordinati
        oP["alphas"] = np.linalg.solve(UU, mapped_points[:, 0])
        
        # beta koeficienti med originalnimi in mapped tockami po y koordinati
        oP["betas"] = np.linalg.solve(UU, mapped_points[:, 1])

        oP["pts"] = original_points

    return oP

if __name__ == "__main__":
    T  = getParameters("affine", rot = 30)
    print("Parametri affine preslikave", T)

    xy = np.array([[0,0], [511, 0], [0, 511], [511, 511]])
    uv = np.array([[0,0], [511,0], [0,511], [255,255]])
    P = getParameters("radial", original_points=xy, mapped_points=uv)
    print("radialna preslikava \n", P)

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

if __name__ == "__main__":
    backgroundValue = 63
    affineImage = transformImage("affine", I, pxDim, np.linalg.inv(T), iBackground=backgroundValue)
    display_image(affineImage, "affina preslikava", gX, gY)

    radialImage = transformImage("radial", I, pxDim, P, iBackground=backgroundValue)
    display_image(radialImage, "radialna slika", gX, gY)

    
    