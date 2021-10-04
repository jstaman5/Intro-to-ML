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

    #initialize cluster centroids randomly
    clusters = np.random.randint(0,255, size=(K,3))

    h, w, c = image.shape
    original_img = image.copy()
    copy = image.copy()
    #max iterations
    iterations = 24

    for it in range(iterations):
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
        
        delta = []
        pixels = []
        u = False
        for i in range(K):
            X, Y, c = np.where(copy == clusters[i])
            points = original_img[X, Y]
            #how many pixels are in each centroid
            print(points.size)
            pixels.append(points.size)
            new_centroid = np.mean(points, axis = 0)
            d = calcDist(new_centroid, clusters[i])
            delta.append(d)
            if(d > 1):
                u = True
            
            if u == False:
                break

            clusters[i] = new_centroid

    return copy, delta, pixels



    
def main(): 
    #read in image
    image = io.imread('baboon.jpg')
    new_image = Kmeans(image, 16)
    #io.imshow(new_image) 
    
    f, axarr = plt.subplots(nrows=2, ncols=2, figsize=(15,4))

    axarr[0].imshow(image)

    K = [4]
    for i, k in enumerate(K):
        new_image, delta, pixels = Kmeans(image, k)

        axarr[i].imshow(new_image)

    
    plt.show()

if __name__ == "__main__":
    main()



'''
#io.imshow(image)
#io.show()
#Dimension of the original image
rows = image.shape[0]
cols = image.shape[1]

#Flatten the image
image = image.reshape(rows*cols, 3)

#Implement k-means clustering to form k clusters
kmeans = KMeans(n_clusters=8)
kmeans.fit(image)

#Replace each pixel value with its nearby centroid
compressed_image = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#Save and display output image
#io.imsave('compressed_image_64.png', compressed_image)
io.imshow(image)
io.imshow(compressed_image)
io.show()
plt.show()
'''