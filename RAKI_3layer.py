# © 2021 Regents of the University of Minnesota. Academic and Research Use Only.

#import print_function
import tensorflow as tf
import scipy.io as sio
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weight_variable(shape,vari_name):                   
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

#### LEANING FUNCTION ####
def learning(ACS_input,target_input,accrate_input,sess):
    
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                  
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])         
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input)) 

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)

    error_norm = tf.norm(input_Target - h_conv3)       
    train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1 
    for i in range(MaxIteration):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 100 == 0:                                                                      
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})    
            print('The',i,'th iteration gives an error',error_now)                             
            
            
        
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),error]  


def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,acc_rate,sess):                
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    return sess.run(h_conv3)                                       

    

#######################################################################
###                                                                 ###
### For convinience, everything are the same with Matlab version :) ###
###                                                                 ###
#######################################################################

###################### Reconstruction Parameters ######################

#### Network Parameters ####
kernel_x_1 = 5
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = 32 
layer2_channels = 8

MaxIteration = 1000
LearningRate = 3e-3

#### Input/Output Data ####
inputData = 'rawdata.mat'
input_variable_name = 'kspace'
resultName = 'RAKI_recon'
recon_variable_name = 'kspace_recon'

######################################################################

kspace = sio.loadmat(inputData)
kspace = kspace[input_variable_name] 
no_ACS_flag = 0;
normalize = 0.015/np.max(abs(kspace[:]))
kspace = np.multiply(kspace,normalize)   

[m1,n1,no_ch] = np.shape(kspace)
no_inds = 1

kspace_all = kspace;
kx = np.transpose(np.int32([(range(1,m1+1))]))                  
ky = np.int32([(range(1,n1+1))])

kspace = np.copy(kspace_all)
mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; 
picks = np.where(mask == 1);                                  
kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  

kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0;  
picks = np.where(mask == 1);                                  
d_picks = np.diff(picks,1)  
indic = np.where(d_picks == 1);

mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;
picks_x = np.where(mask_x == 1);
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

if np.size(indic)==0:    
    no_ACS_flag=1;       
    print('No ACS signal in input data, using individual ACS file.')
    matfn = 'ACS.mat'   
    ACS = sio.loadmat(matfn)
    ACS = ACS['ACS']     
    normalize = 0.015/np.max(abs(ACS[:])) 
    ACS = np.multiply(ACS,normalize*scaling)

    kspace = np.multiply(kspace,scaling)
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
else:
    no_ACS_flag=0;
    print('ACS signal found in the input data')
    indic = indic[1][:]
    center_start = picks[0][indic[0]];
    center_end = picks[0][indic[-1]+1];
    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)

acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z*2

name_weight = resultName + ('_weight_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))
name_image = resultName + ('_image_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))

existFlag = os.path.isfile(name_image)

w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32)    

b1_flag = 0;
b2_flag = 0;                       
b3_flag = 0;

if (b1_flag == 1):
    b1_all = np.zeros([1,1, layer1_channels,no_channels]);
else:
    b1 = []

if (b2_flag == 1):
    b2_all = np.zeros([1,1, layer2_channels,no_channels])
else:
    b2 = []

if (b3_flag == 1):
    b3_all = np.zeros([1,1, layer3_channels, no_channels])
else:
    b3 = []

target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1); 
target_x_end = np.int32(ACS_dim_X - target_x_start -1); 

time_ALL_start = time.time()

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) 
ACS = np.float32(ACS)  

target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

print('go!')
time_Learn_start = time.time() 

errorSum = 0;
config = tf.ConfigProto()


for ind_c in range(ACS_dim_Z):

    sess = tf.Session(config=config)
    # set target lines
    target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
    print('learning channel #',ind_c+1)
    time_channel_start = time.time()
    
    for ind_acc in range(acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

    # learning

    [w1,w2,w3,error]=learning(ACS,target,acc_rate,sess) 
    w1_all[:,:,:,:,ind_c] = w1
    w2_all[:,:,:,:,ind_c] = w2
    w3_all[:,:,:,:,ind_c] = w3                               
    time_channel_end = time.time()
    print('Time Cost:',time_channel_end-time_channel_start,'s')
    print('Norm of Error = ',error)
    errorSum = errorSum + error

    sess.close()
    tf.reset_default_graph()
    
time_Learn_end = time.time();
print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min')
sio.savemat(name_weight, {'w1': w1_all,'w2': w2_all,'w3': w3_all})  


kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)

kspace = np.copy(kspace_all)

over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
kspace_und = kspace
kspace_und[:,over_samp,:] = 0;
[dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_recon = kspace_und_re

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3 ; 

for ind_c in range(0,no_channels):
    print('Reconstruting Channel #',ind_c+1)
    
    sess = tf.Session(config=config) 
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    
    # grab w and b
    w1 = np.float32(w1_all[:,:,:,:,ind_c])
    w2 = np.float32(w2_all[:,:,:,:,ind_c])     
    w3 = np.float32(w3_all[:,:,:,:,ind_c])

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];                
        
    res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,acc_rate,sess) 
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
    
    for ind_acc in range(0,acc_rate-1):

        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;             
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

    sess.close()
    tf.reset_default_graph()
    
kspace_recon = np.squeeze(kspace_recon)

kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); 


if no_ACS_flag == 0: 
    kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
    print('ACS signal has been putted back')
else:
    print('No ACS signal is putted into k-space')

kspace_recon_all[:,:,:] = kspace_recon_complex; 

for sli in range(0,no_ch):
    kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])

rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
sio.savemat(name_image,{recon_variable_name:kspace_recon_complex})

time_ALL_end = time.time()
print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
print('Error Average in Training is ',errorSum/no_channels)



