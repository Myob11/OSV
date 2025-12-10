from matplotlib import pyplot as plt
import numpy as np 



def display_image(image, title="", cmap = None):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
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

path = r"vaja1\data\lena-gray-410x512-08bit.raw"
type = np.uint8
size1 = (410,512)
size2 = (512,410)
im = load_image(path, size1, type)
display_image(im, path, cmap='RdBu')


## dodatno gradivo
def saveImage(iImage, iPath, iType):

    fid = open(iPath, "wb")
    oImage = np.array(iImage, dtype=iType)
    # zapisemo v Fortranskem nacinu, da lahko za nalaganje te datoteke uporabimo naso loadImage funkcijo
    fid.write(oImage.tobytes(order="F"))

    fid.close()
    return



 
