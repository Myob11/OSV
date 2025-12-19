import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import load_image, display_image   

## 1. NALOGA
#%%
if __name__ == "__main__":
    g = load_image(r"vaja10\data\pattern-236x330-08bit.raw", [236, 330], np.uint8)
    display_image(g, "OG slika")

#%%

def computeDFT2 ( iMatrix , inverse = False ):
    
    # vhodna matrika je v nasem primeru slika
    N, M = iMatrix.shape

    n = np.arange(N).reshape(1, -1)
    m = np.arange(M).reshape(1, -1)

    # naredimo 2 fourier transforma v horizontalni in vertikalni smeri
    WN = 1 / np.sqrt(N) * np.exp(-1j * 2 * np.pi / N) ** (n.T @ n) # np.exp je e^(kar je v oklepaju)
    WM = 1 / np.sqrt(M) * np.exp(-1j * 2 * np.pi / M) ** (m.T @ m) # n.t @ n da dobimo absolutno vrednost

    if inverse:
        # konjugacija v praksi je zamenjava predznaka imag vrednosti
        # v nasem primeru e ** (..)
        WN = np.conj(WN)
        WM = np.conj(WM)

    oMatrix = WN @ iMatrix @ WM

    return oMatrix

if __name__ == "__main__":
    G = computeDFT2(g)
    gR = computeDFT2(G, inverse=True)
    display_image(G.real, "transformirana slika")
    display_image(gR.real, "rekonstruirana slika")

#%%
def analyzeDFT2 (iMatrix , iOperations , iTitle =""):

    # mutability: enako kot da bi klicali .copy
    # da ne spreminjamo ponesrec vhodne slike
    oMatrix = np.array(iMatrix)

    for operation in iOperations:

        if operation == "amplitude":
            oMatrix = np.abs(oMatrix)

        elif operation == "phase":
            # razdeli matriko na eno dimenzijo
            # np.angle poracuna kote -pi do pi na nasi vhfoni matriki
            np.unwrap(np.angle(oMatrix))

        elif operation == "ln":
            # da se izognemo log(0) dodamo zelo majhno vrednost zraven
            # np.log po defaultu je ln 
            oMatrix = np.log(oMatrix + 1e-10)

        elif operation == "log":
            oMatrix = np.log10(oMatrix + 1e-10)

        elif operation == "scale":
            # najprej odstejemo min vrednost da se nase def obmocje zacne z 0
            oMatrix -= oMatrix.min()
            # deljimo z max vrednostjo, normaliziramo vrednosti med 0 in 1
            oMatrix /= oMatrix.max()
            # mnozimo z max 8bit vrednostjo, da je nase obmocje med 0 in 255
            oMatrix *= 255
            # shramimo vrednosti kot uint8, da se znebimo decimalnih vrednosti
            oMatrix = oMatrix.astype(np.uint8)

        elif operation == "center":
            N, M = oMatrix.shape
            # najdemo center matrike
            n_c, m_c = int((N - 1) / 2), int((M - 1) / 2)
            # izluscimo 4 kvadrante vhodne matrike
            A = oMatrix[:n_c, :m_c]
            B = oMatrix[n_c:, :m_c]
            C = oMatrix[n_c:, m_c:]
            D = oMatrix[:n_c, m_c:]

            # sestacimo jih skupaj, jih zavrtimo

            upper = np.hstack((C, B))
            lower = np.hstack((D, A))

            oMatrix = np.vstack((upper, lower))

        elif operation == "display":
            plt.figure()
            plt.imshow(
                oMatrix,
                aspect="equal",
                cmap=plt.cm.gray
            )
            plt.title(iTitle)
            # plt.show()

        else:
            raise NotImplementedError("Nedefinirana operacija: ", operation)
             
    return oMatrix  

if __name__ == "__main__":
    # amplitudni diagram, sredina predstavlja nizke frekv, proti robu pa vedno visje
    # za amplitudo hocemo diagram centrirati, da bodo visoke frekv na kupu, log skala
    # vrednosti v frekv prostoru, scale in display za lep prikaz
    analyzeDFT2(
        G,
        iOperations=["amplitude", "center", "log", "scale", "display"],
        iTitle = "Amplituda (log skaliranje)"
    )

    analyzeDFT2(
        G,
        iOperations=["phase", "scale", "display"],
        iTitle = "Faza"
    )

