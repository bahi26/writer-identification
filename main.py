import os
import cv2
import image_cropping
import lbp
import preprocessing
import svm_knn
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from skimage import data


def main():

    results=[]
    #precentage=0;
    extension='.jpg'
    path = 'F:\\handwritting\\Test2'

    for foldername in os.listdir(path):
        training_features = []
        labels=[]

        for t in range(1,4):
            for i in range(1,3):
                training_image= cv2.imread(path+'\\'+foldername+'\\'+str(t)+'\\'+str(i)+extension)
                texture=[]
                #print(foldername)
                cropped=cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)

                #cropped = image_cropping.crop(training_image)
                texture= preprocessing.preprocess(cropped)
                for y in range(0, len(texture)):
                    labels.append(t)
                for element in texture:
                    #print(element.shape)
                    lbp2 = local_binary_pattern(element, 16, 1, method='nri_uniform')
                    (hist, _) = np.histogram(lbp2[np.where(lbp2<175)],bins=256,range=(0, 256))
                    eps = 1e-7
                    # normalize the histogram
                    hist = hist.astype("float")
                    hist /= (hist.sum() + eps)

                    training_features.append(hist)





        test_image= cv2.imread(path + '\\' + foldername + '\\' + 'test'+extension)
        #cropped_test = image_cropping.crop(test_image)
        texture_test=[]
        cropped_test=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        texture_test = preprocessing.preprocess(cropped_test)

        votes=[]
        test_features=[]
        for element_t in texture_test:


            lbp2 = local_binary_pattern(element_t, 16, 1, method='nri_uniform')
            (hist, _) = np.histogram(lbp2[np.where(lbp2<175)],
                                     bins=256,
                                     range=(0, 256))
            eps = 1e-7
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            test_features.append(hist)




       # votes=svm_knn.knn_classifier(training_features,labels,test_features)
        votes=svm_knn.myknn(training_features,labels,test_features)
        counts=np.bincount(votes)
        writer_detected=np.argmax(counts)
        print('votes',votes)
        print('result',writer_detected)



if __name__== "__main__":
    main()