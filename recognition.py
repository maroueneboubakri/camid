'''
__author__ = "Marouene Boubakri"
__copyright__ = "Copyright (C) 2017 Marouene"
__license__ = "Public Domain"
__version__ = "1.0"
'''

import numpy as np
import Image
import math 
import pywt
import copy
import scipy.io
import operator
import PIL

import glob, os


def im2col(Im, block, style='sliding'):
    bx, by = block
    Imx, Imy = Im.shape
    Imcol = []
    for j in range(0, Imy):
        for i in range(0, Imx):
            if (i + bx <= Imx) and (j + by <= Imy):
                Imcol.append(Im[i:i + bx, j:j + by].T.reshape(bx * by))
            else:
                break
    return np.asarray(Imcol).T
   

def im2col_sliding(A, size):  
    dy, dx = size  
    xsz = A.shape[1] - dx + 1
    ysz = A.shape[0] - dy + 1
    R = np.empty((xsz * ysz, dx * dy))
      
    for i in xrange(ysz):  
        for j in xrange(xsz):  
            R[i * xsz + j, :] = A[i:i + dy, j:j + dx].ravel()
    return R  

def col2im_sliding(B, block_size, image_size):
    m,n = block_size
    mm,nn = image_size
    #return B.reshape(nn-n+1,mm-m+1).T 
      # Or simply B.reshape(nn-n+1,-1).T
    return B.reshape(mm-m+1,nn-n+1,order='F')
    
    
def _denoise_band(X, wavelet, levels, sigma):
    if sigma is None:
        sigma = 5    
        
    noised_coeffs = []
    denoised_coeffs = []
    decomp = pywt.wavedec2(X, wavelet, mode='symmetric', level=levels)
    approximation = decomp[0].ravel()
    noised_coeffs.append(approximation)       
    denoised_coeffs.append(approximation)

    window_sizes = [3,5,7,9]
    for i, all_coeff in enumerate(decomp[1:]):
        for j, coeff in enumerate(all_coeff):            
            cHTS = coeff.shape
            varE = np.matlib.repmat(float(0), len(window_sizes), cHTS[0]*cHTS[1])
            for k,w in enumerate(window_sizes):    
                p = int(np.floor(w/2))    
                buf = np.pad(coeff, [p,p], 'symmetric')
                bufS = buf.shape
                c = im2col(buf, [w,w], 'sliding')
                csum = np.sum(c**2,axis=0)
                csum = np.divide(csum,w**2);
                csum = np.subtract(csum,sigma**2)
                varE[k,:] = np.maximum(0, csum)

            var_min = np.amin(varE, axis = 0)            
            var_est = col2im_sliding(var_min, (w,w), bufS)
            noised_coeffs.append(np.ravel(coeff, order='F').T)            
            coeff *= np.divide(var_est,var_est+(sigma**2))
            denoised_coeffs.append(np.ravel(coeff, order='F').T)

    rec = pywt.waverec2(decomp, wavelet)   

    rows, cols = X.shape
    if X.shape != rec.shape:
        rows_mod = rows % 2
        cols_mod = cols % 2
        return rec[rows_mod:, cols_mod:], noised_coeffs, denoised_coeffs
    else:
        return rec, noised_coeffs, denoised_coeffs

def denoise(X, wavelet='db8', levels=4, sigma=5):

    out = np.zeros(X.shape, dtype=float)
    
    noised_coeffs_pack = []
    denoised_coeffs_pack = []
    if X.ndim == 3:        
        bands = X.shape[2]
        for b in range(bands):
            denoised_band, noised_coeffs, denoised_coeffs = _denoise_band(X[..., b], wavelet, levels, sigma) 
            noised_coeffs_pack.append(noised_coeffs)
            denoised_coeffs_pack.append(denoised_coeffs)
            
            out[:, :, b] = denoised_band
                    
        noised_coeffs_pack = np.array(noised_coeffs_pack)
        noised_coeffs_pack = np.concatenate([np.concatenate(x) for x in zip(*noised_coeffs_pack)])

        denoised_coeffs_pack = np.array(denoised_coeffs_pack)
        denoised_coeffs_pack = np.concatenate([np.concatenate(x) for x in zip(*denoised_coeffs_pack)])       
        
        sensor_pattern_noise = np.subtract(noised_coeffs_pack , denoised_coeffs_pack)        

    else:
        denoised_band, noised_coeffs, denoised_coeffs = _denoise_band(X, wavelet, levels, sigma)
        sensor_pattern_noise = np.subtract(noised_coeffs , denoised_coeffs)
        out[:] = denoised_band

    alpha = 7
    enhanced_sensor_pattern_noise = np.zeros_like(sensor_pattern_noise)       
    
    enhance = lambda n: math.exp((-0.5*(n**2))/ (alpha ** 2)) if n <=0 else (-math.exp((-0.5*(n**2))/(alpha** 2)))
    
    vec_enhance = np.vectorize(enhance)
    
    enhanced_sensor_pattern_noise = vec_enhance(sensor_pattern_noise)
     
    return out, sensor_pattern_noise, enhanced_sensor_pattern_noise



def calculate_rspn(bg_img_path):
    imgs =  glob.glob(bg_img_path+"/*.jpg")
    running_sum = None
    count = 0
    for file in imgs:
        print "[+] ("+str(count+1)+") "+file        
        im = Image.open(file)    
        new_width = 512
        new_height = 512    
        width, height = im.size
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2        
        im =  im.crop((left, top, right, bottom))
        imdata, spn, espn = denoise(np.asarray(im), sigma=5)
        count += 1        
        if running_sum is None:
            running_sum = spn
        else:
            running_sum = np.add(running_sum, spn)
    average = np.divide(running_sum,count)
    scipy.io.savemat('rspn.mat', {'rspn': average})
    return average
            


def detect_image(image_path, camera_noise):
    im = Image.open(image_path)
    
    new_width = 512
    new_height = 512
    
    width, height = im.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))
    im.load()
    
    out, image_noise, image_enhanced_noise = denoise(np.asarray(im), wavelet='db8', levels=4, sigma=5)    
    
    camera_noise_average = np.average(camera_noise)
    camera_noise -= camera_noise_average
    camera_noise_norm = np.sqrt(np.sum(camera_noise * camera_noise))

    image_noise_average = np.average(image_noise)
    image_noise -= image_noise_average
    image_noise_norm = np.sqrt(np.sum(image_noise * image_noise))
  
    return np.sum(camera_noise * image_noise) / (camera_noise_norm * image_noise_norm)  


def detect_images(images_path, camera_noise):
    imgs =  glob.glob(images_path+"/*.jpg")
    for img in imgs:
        print "[+] "+ img + ", ",
        print detect_image(img, camera_noise)
        


if __name__ == "__main__":
    #avg =  calculate_rspn('./bg')
    #quit()
    camera_noise = scipy.io.loadmat('rspn.mat')['rspn']
    detect_images('./sample', camera_noise)



