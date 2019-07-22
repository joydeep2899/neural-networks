import skimage.data
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot

def conv_(img,convfilter):
  filter_size=convfilter.shape[1]
  result=np.zeros((img.shape))  
  for r in (np.uint16(np.arange(filter_size/2.0,img.shape[0]-filter_size/2.0+1))):
       for c in (np.uint16(np.arange(filter_size/2.0,img.shape[1]-filter_size/2.0+1))):
           
         curr_region=img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
         curr_result=curr_region * convfilter
         conv_sum=np.sum(curr_result)
         result[r,c]=conv_sum
  final_result=result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0),np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
  return final_result

def conv(img,convfilter):
   if len(img.shape) >2 or len(convfilter.shape) >3:
           if img.shape[-1] != convfilter.shape[-1]:
              print("the number of channels in image and fitler must match")
              sys.exit()
   if convfilter.shape[1] != convfilter.shape[2]:
              print("the filter must be a square filter")
              sys.exit()
   if convfilter.shape[1]%2==0:
              print("the filter must be odd in shape")
              sys.exit()
   feature_maps= np.zeros((img.shape[0]-convfilter.shape[1]+1,img.shape[1]-convfilter.shape[2]+1,convfilter.shape[0]))
            
   for filter_num in range(convfilter.shape[0]):
               curr_filter=convfilter[filter_num,:]
                 
               if len(curr_filter.shape)>2:
                  conv_map= conv_(img[:,:,0],curr_filter[:,:,0])
                  
                  for ch_num in range(1,curr_filter.shape[-1]):
                      conv_map=conv_map+conv_(img[:,:,ch_num],curr_filter[:,:,ch_num])
               else :
                  conv_map=conv_(img,curr_filter)
               feature_maps[:,:,filter_num]=conv_map
   return feature_maps
 
def pooling(feature_map,size=2,stride=2):
    pool_out=np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1),np.uint16((feature_map.shape[1]-size+1)/stride+1),feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]): 
        r2=0
        for r in (np.arange(0,feature_map.shape[0]-size+1,stride)):  
                 c2=0
                 for c in (np.arange(0,feature_map.shape[1]-size+1,stride)):
                    pool_out[r2,c2,map_num]=np.max([feature_map[r:r+size,c:c+size]])
                    c2=c2+1
                 r2=r2+1
    return  pool_out
def relu(feature_map):
    relu_out=np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in (np.arange(0,feature_map.shape[0])): 
             for c in (np.arange(0,feature_map.shape[1])):
                 relu_out[r,c,map_num]=np.max([feature_map[r,c,map_num],0])
    return relu_out


img = skimage.io.imread("dog.jpeg")
# Converting the image into gray.
img = skimage.color.rgb2gray(img)
# First conv layer
#l1_filter = numpy.random.rand(2,7,7)*20 #
l1_filter = np.zeros((2,3,3))
l1_filter[0, :, :] = np.array([[[-1,  0,  1],
                                   [-1,   0,  1],
                                   [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1,   1,  1],
                                   [0,   0,  0],
                                   [-1, -1, -1]]])
print("\n**Working with conv layer 1**")
l1_feature_map = conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
print("**End of conv layer 1**\n")


l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 2**")
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
print("\n**ReLU**")
l2_feature_map_relu = relu(l2_feature_map)
print("\n**Pooling**")
l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**\n")

l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 3**")
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
print("\n**ReLU**")
l3_feature_map_relu = relu(l3_feature_map)
print("\n**Pooling**")
l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")

l4_filter = np.random.rand(5, 7, 7, l3_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 3**")
l4_feature_map = conv(l3_feature_map_relu_pool, l4_filter)
print("\n**ReLU**")
l4_feature_map_relu = relu(l4_feature_map)
print("\n**Pooling**")
l4_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")





# Graphing results
fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
matplotlib.pyplot.close(fig0)

# Layer 1
fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")

ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")

ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")

matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)

# Layer 2
fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=3)
ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])
ax2[0, 0].set_title("L2-Map1")

ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])
ax2[0, 1].set_title("L2-Map2")

ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
ax2[0, 2].get_xaxis().set_ticks([])
ax2[0, 2].get_yaxis().set_ticks([])
ax2[0, 2].set_title("L2-Map3")

ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])
ax2[1, 0].set_title("L2-Map1ReLU")

ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])
ax2[1, 1].set_title("L2-Map2ReLU")

ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
ax2[1, 2].get_xaxis().set_ticks([])
ax2[1, 2].get_yaxis().set_ticks([])
ax2[1, 2].set_title("L2-Map3ReLU")
ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])
ax2[2, 0].set_title("L2-Map1ReLUPool")

ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])
ax2[2, 1].set_title("L2-Map2ReLUPool")

ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
ax2[2, 2].get_xaxis().set_ticks([])
ax2[2, 2].get_yaxis().set_ticks([])
ax2[2, 2].set_title("L2-Map3ReLUPool")

matplotlib.pyplot.savefig("L2.png", bbox_inches="tight")
matplotlib.pyplot.close(fig2)

# Layer 3
fig3, ax3 = matplotlib.pyplot.subplots(nrows=1, ncols=3)
ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
ax3[0].get_xaxis().set_ticks([])
ax3[0].get_yaxis().set_ticks([])
ax3[0].set_title("L3-Map1")

ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
ax3[1].get_xaxis().set_ticks([])
ax3[1].get_yaxis().set_ticks([])
ax3[1].set_title("L3-Map1ReLU")

ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax3[2].get_xaxis().set_ticks([])
ax3[2].get_yaxis().set_ticks([])
ax3[2].set_title("L3-Map1ReLUPool")

matplotlib.pyplot.savefig("L3.png", bbox_inches="tight")
matplotlib.pyplot.close(fig3)
