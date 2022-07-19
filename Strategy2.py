#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('9095') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[29]:


overall_losses = []

for ks in range(2,11,1):

    #input values
    points = data
    iterations = 20
    loops = 0
    n = len(points)
    k = ks
    centroids = numpy.zeros([k,2])
    centroids[0,:] = points[random.randint(0,n-1)]

    #max distance initial centroid generation
    centroid_losses = numpy.zeros([n,(k-1)]) #array for the distance from each point to each centroid
    points_temp = points
    for i in range(k-1): #the row in centroids being checked

      #finds distance from the centroid to each point
      for index, point in enumerate(points_temp):
        distance = ((centroids[i,0] - point[0])**2 + (centroids[i,1] - point[1])**2)**0.5 #calc dist
        centroid_losses[index,i] = distance

      sums = numpy.mean(centroid_losses, dtype = numpy.longdouble,axis = 1) #sums the centroid distances for each point
      new_centroid_index = numpy.argmax(sums, axis = 0) #finds the datapoint index at max combined distance
      centroids[i+1] = points_temp[new_centroid_index]
      points_temp = numpy.delete(points_temp,new_centroid_index, axis = 0)
      centroid_losses = numpy.delete(centroid_losses,new_centroid_index, axis = 0)

    #actual k means begins
    while loops < iterations:

      #finds distance from each centroid to each point
      losses = numpy.empty([n,k]) #holds distances from each centroid, one col per troid
      for count, mean in enumerate(centroids):
        distances = numpy.empty([n,1]) #holds distances for one centroid

        for index, point in enumerate(points):
          distance = (mean[0] - point[0])**2 + (mean[1] - point[1])**2 #calc dist
          distances[index,0] = distance #assigns dist val to list

        losses[:,count] = distances[:,0] #assigns column of dists to losses

      #assign points to closest centroid
      assignment = np.empty([n,2]) #one column for centroid index assignment, one for distance
      assignment[:,0] = numpy.argmin(losses, axis = 1) #index of min centroid distance
      assignment[:,1] = numpy.amin(losses, axis = 1) #value of min distance

      loss = numpy.sum(assignment[:,1], axis = 0) # total training loss

      #recalculate centroids
      new_centroids = numpy.empty([k,2])
      for i, centroid in enumerate(new_centroids):
        cluster = numpy.where(assignment[:,0] == i)
        new_centroids[i,:] = numpy.mean(points[cluster], axis = 0)#col row access might be weird here

      centroids = new_centroids #possible off by one here with the loss being for the old centroids
      loops +=1
    overall_losses.append(loss)


# In[8]:


plt.scatter(points[:,0],points[:,1],c=assignment[:,0])
plt.scatter(centroids[:,0], centroids[:,1], marker = "P")
plt.title("Centroids and Clusters for Strategy 2, k = 4")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[31]:


plt.scatter(range(2,11,1),overall_losses)
plt.title("Cost vs. K for Strategy 2")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squared Error (cost)")
plt.show()


# In[27]:


len(overall_losses)


# In[ ]:




