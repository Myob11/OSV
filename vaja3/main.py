import numpy as np
from OSV_lib import load_image, display_image
from OSV_lib import computeHistogram, displayHistogram

# koordniatni sistem vhodne slike je 3x2
# koordinatni sistem izhodne pa 7x3

# dx = X-1 / M-1
# dx = 2 / 6

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


if __name__ == '__main__':

    I = load_image(r"vaja3\data\pumpkin-200x152-08bit.raw", [200,152], np.uint8)
    display_image(I, "Originalna slika")
    oSize = [I.shape[1]*2, I.shape[0]*2] # povečamo sliko x2 po X in x2 po Y
    
    I0 = InterpolateImage(I, oSize, 0)
    print(I0.shape)
    display_image(I0, 'interpolirana slika (red 0)')

    I1 = InterpolateImage(I, oSize, 1)
    print(I0.shape)
    display_image(I0, 'interpolirana slika (red 1)')

    Ipatch = I[29:29+50, 74:74+65]
    display_image(Ipatch, 'Koscek originalne slike')
    h, _,_, l = computeHistogram(Ipatch)
    displayHistogram(h,l, 'Histogram (orig)')

    Ipatch0 = InterpolateImage(Ipatch, [600, 300], 0)
    Ipatch1 = InterpolateImage(Ipatch, [600, 300], 1)
    display_image(Ipatch0, 'koscek (red 0)')
    display_image(Ipatch1, 'koscek (red 1)')
    h0, _, _, l0 = computeHistogram(Ipatch0)
    displayHistogram(h0,l0, 'Histogram (red 0)')
    h1, _, _, l1 = computeHistogram(Ipatch1)
    displayHistogram(h1,l1, 'Histogram (red 1)')


if __name__ == '__main__':
    gridX = np.arange(40)
    gridY = np.arange(80)
    display_image(Ipatch, 'Test displayImage', gridX, gridY)

## dodatno gradivo
"""
Napišite funkcijo za decimacijo slike:
kjer vhodni argument iImage predstavlja decimacijsko sliko, 
iKernel jedro c(i, j) digitalnega filtra velikosti N × N, 
iLevel pa je celoštevilski nivo piramidne decimacije, 
medtem ko izhodni argument oImage predstavlja decimirano sliko.
Decimirajte dano sliko z danima jedroma c(i, j) digitalnega filtra 
velikosti M = 1 in M = 2 ter prikažite rezultate na nivoju piramidne decimacije iLevel = 2.

"""
#%%
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

if __name__ == '__main__':
    
    Kernel_1 = np.array([
        [1/16, 1/8, 1/16],
        [1/8,  1/4, 1/8],
        [1/16, 1/8, 1/16]
                         ])
    
    Kernel_2 = np.array([
        [1/400, 1/80, 1/50, 1/80, 1/400],
        [1/80,  1/16, 1/10, 1/16, 1/80],
        [1/50,  1/10, 4/25, 1/10, 1/50],
        [1/80,  1/16, 1/10, 1/16, 1/80],
        [1/400, 1/80, 1/50, 1/80, 1/400]
    ])

    display_image(I, "OG")
    print("OG shape", I.shape)
    NS = decimateImage(I, Kernel_1, 1)
    display_image(NS, "decimirana slika")

    NS = decimateImage(I, Kernel_1, 2)
    display_image(NS, "decimirana slika 1, level 2")

    NS = decimateImage(I, Kernel_2, 1)
    display_image(NS, "decimirana slika kernel 2")

    NS = decimateImage(I, Kernel_2, 2)
    display_image(NS, "decimirana slika kernel 2, level 2")