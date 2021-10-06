#Jared Staman
#CS 425: Assignment 3


from skimage import io
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans 

def calcDist(point, centroid):
    r, g, b = point
    c_r, c_g, c_b = centroid

    return math.sqrt( (r-c_r)**2 + (g-c_g)**2 + (b-c_b)**2)

def Kmeans(image, K):
    np.seterr(divide='ignore', invalid='ignore')
    #initialize cluster centroids randomly
    clusters = np.random.randint(0,255, size=(K,3))
    delta = []
    h, w, c = image.shape
    original_img = image.copy()
    copy = image.copy()
    #max iterations
    iterations = 24
    stop = False
    for it in range(iterations):
        if(stop == True):
            break

        copy = image.copy()
        #loop through every pixel in image
        for i in range(h):
            for j in range(w):
                #set minimum distance to start at infinity
                min = float("inf")
                pnt = copy[i][j]
                #loop through the clusters
                #for centroid in clusters:
                    #d = calcDist(pnt, centroid)
                d=np.sqrt(np.sum((clusters-pnt)**2,axis=1))
                c=np.argmin(d)
                copy[i][j]=clusters[c]
                    #if a point is more similar in color to another centroid, then change it
                    #if(d < min):
                       # min = d
                       # copy[i][j] = centroid
        
        #delta = []
        tmp = []
        pixels = []
        u = False
        for i in range(K):
            X, Y, c = np.where(copy == clusters[i])
            points = original_img[X, Y]
            #how many pixels are in each centroid
            #print(np.size(X))
            pixels.append(np.size(X))
            new_centroid = np.mean(points, axis = 0)
            #print(new_centroid)
            d = calcDist(new_centroid, clusters[i])
            clusters[i] = new_centroid
            #print(d)
            tmp = []
            if(math.isnan(d)):
                tmp.append(0.0)
            else:
                tmp.append(d)
            '''
            #print(tmp)
            if(i == K - 1 ):
                d2 = sum(tmp) / len(tmp)
                delta.append(d2)
                if (d2 < 1):
                    stop = True
                    print(f'iterations: {it}')
                print(delta)
            '''
            if(i == K - 1):
                if(math.isnan(sum(tmp) / len(tmp))):
                    delta.append(0.0)
                else:
                    delta.append( sum(tmp) / len(tmp))
                #print(delta)
        
            #delta.append(d)
            if(d > 1):
                u = True
            
            if u == False:
                #print(f'distance: {d}')
                print(f'iterations: {it}')
                stop = True
            
            if i == K - 1 and stop == True:
                break

            #clusters[i] = new_centroid

    return copy, pixels ,delta



    
def main(): 
    
    #I loop through manually, testing each part for each picture myself to get the graphs on the report
    #For this project, I did run into the issue of nan values cutting some K-means algorithms short
    #so I had to run my code multiple times to get good examples

    #GRAPHING THE LINE GRAPHS
    '''
    image = io.imread('smokey.jpg')
    iterations = []
    new_image, pixels, delta = Kmeans(image,4)
    for idx, val in enumerate(delta):
        iterations.append(idx)
    plt.axis((0,23, 0, 45))
    plt.plot(iterations, delta, label = '4')

    iterations = []
    new_image, pixels, delta = Kmeans(image,16)
    for idx, val in enumerate(delta):
        iterations.append(idx)
    plt.axis((0,23, 0, 45))
    plt.plot(iterations, delta, label =  '16')

    iterations = []
    new_image, pixels, delta = Kmeans(image,32)
    for idx, val in enumerate(delta):
        iterations.append(idx)
    plt.axis((0,23, 0, 45))
    plt.plot(iterations, delta, label = '32')
    plt.legend()
    plt.show()
    '''

    
    #DISPLAYING 2x2 PICTURES
    '''
    image = io.imread('baboon.jpg')
    f, axarr = plt.subplots(nrows = 2, ncols = 2, figsize = (15,4))
    
    axarr[0,0].imshow(image)
    axarr[0,0].set_title("Original Image")
    new_image, delta, pixel = Kmeans(image, 4)
    axarr[0,1].imshow(new_image)
    axarr[0,1].set_title("Image Obtained for K = 4")
    new_image, delta, pixel = Kmeans(image, 16)
    axarr[1,0].imshow(new_image)
    axarr[1,0].set_title("Image Obtained for K = 16")
    new_image, delta, pixel = Kmeans(image, 32)
    axarr[1,1].imshow(new_image)
    axarr[1,1].set_title("Image Obtained for K = 32")
    f.tight_layout()
    plt.show()
    '''
    

    #GRAPHING THE HISTOGRAMS
    
    image = io.imread('baboon.jpg')
    new_image, pixel, delta = Kmeans(image, 4)
    x = [x for x in range(5) if x!= 0]
    #x2 = [x for x in range(17) if x!= 0]
    #x3 = [x for x in range(33) if x!= 0]
    print(pixel)
    total_pixels = sum(pixel)
    pixel_percentage = []
    for idx, p in enumerate(pixel):
        pixel_percentage.append( (p / total_pixels) * 100)
    plt.subplot(1,3,1)
    plt.bar(x, pixel_percentage)
    plt.ylabel("Pixels in Cluster")
    plt.xlabel("K = 4")
    plt.ylim((0,50))
    plt.locator_params(axis="both", integer = True)

    new_image, pixel, delta = Kmeans(image, 16)
    x2 = [x for x in range(17) if x!= 0]
    print(pixel)
    total_pixels = sum(pixel)
    pixel_percentage = []
    for idx, p in enumerate(pixel):
        pixel_percentage.append( (p / total_pixels) * 100)
    plt.subplot(1,3,2)
    plt.bar(x2,pixel_percentage)
    plt.xlabel("K = 16")
    plt.title("Cluster Distributions")
    plt.ylim((0,50))
    plt.locator_params(axis="both", integer = True)
    
    new_image, pixel, delta = Kmeans(image, 32)
    x3 = [x for x in range(33) if x!= 0]
    print(pixel)
    total_pixels = sum(pixel)
    pixel_percentage = []
    for idx, p in enumerate(pixel):
        pixel_percentage.append( (p / total_pixels) * 100)
    plt.subplot(1,3,3)
    plt.bar(x3,pixel_percentage)
    plt.xlabel("K = 32")
    plt.ylim((0,50))
    plt.locator_params(axis="both", integer = True)
    plt.show()
    


    

if __name__ == "__main__":
    main()

