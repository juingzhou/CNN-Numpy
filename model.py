import numpy as np
import sys

# ReLU层
def relu(feature_map):
    # 实现relu激活函数
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max(feature_map[r, c, map_num], 0)
    return relu_out


def pooling(feature_map, kernel=2, stride=2):
    pool_out = np.zeros(np.uint16((feature_map.shape[0] - kernel)/stride + 1),
                        np.uint16((feature_map.shape[1] - kernel)/stride + 1),
                        feature_map.shape[-1])
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0]-kernel+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-kernel+1, stride):
                pool_out[r, c, map_num] = np.max(feature_map[r:r+kernel, c:c+kernel, map_num], 0)
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out


def conv(img, conv_filter):
    if (len(img.shape) > 2) or (len(conv_filter.shape) >3):
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    
    if conv_filter.shape[1]%2 == 0:
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()
    
    feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                            img.shape[1]-conv_filter.shape[1]+1, 
                            conv_filter.shape[0]))
    
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]

        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in np.uint16(np.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                          np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result