import cv2
import numpy as np

def LBPFeature(image3, no_blocks=1):
    NoBlocks=no_blocks
        #print(NoBlocks)
    nRows, nColumns = image3.shape
    histo = [0]*256
    matrix = []
    #nColumns=int (nColumns/100)
    for z in range(0, NoBlocks):
        image=image3
        for x in range(1, nRows - 1):
            for y in range(1, nColumns - 1):
                str = ""
                if (image[x - 1][y - 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x - 1][y] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x - 1][y + 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x][y + 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x + 1][y + 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x + 1][y] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x + 1][y - 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                if (image[x][y - 1] >= image[x][y]):
                    str += '1'
                else:
                    str += '0'
                yasmine=int(str, 2)
                histo[yasmine]+=1
                #if x%30==1 and y==1:
                 #   print(x)
                    #print(histo[yasmine])

                #matrix.append(int(str, 2))
    #print(len(matrix))
    #print (histo)
    return histo
