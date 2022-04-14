import numpy as np
import cv2 as cv


def computeKmeans(image):
    Z = image.reshape((-1, 3))  # Redimensionando imagem (-1, 3) altura e largura
    # Critério para interrupção = o epslon garante a diferença entre os conjuntos / Critério de máximo de iterações
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8 # Número de cores
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Convertendo de float para inteiro sem sinal 8 bits
    center = np.uint8(center)
    # Colorindo imagem com Centroide
    res = center[label.flatten()]  # PEgando matriz quadrada e montando vetor
    res2 = res.reshape(image.shape)
    cv.imshow('Res', image)
    cv.imshow('res2', res2)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image = cv.imread('nwh2.jpg')
    computeKmeans(image)
