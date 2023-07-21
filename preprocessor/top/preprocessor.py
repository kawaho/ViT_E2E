import numpy as np
import pyarrow.parquet as pq

from tqdm import tqdm
import torch
import math
import os
import cv2
import time
import multiprocessing

def generate(pf, path, which_file, mean_std):
    sum_channel, sum2_channel, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0

    batch_size = pf.num_row_groups
    nchunks = math.ceil(pf.num_row_groups/batch_size)
    record_batch = pf.iter_batches(batch_size=batch_size)
    for which_batch in range(nchunks):
        batch = next(record_batch)

        sum_channel_tmp, sum2_channel_tmp, size_tmp = process(batch, path, which_file, which_batch, mean_std)

        sum_channel+=sum_channel_tmp
        sum2_channel+=sum2_channel_tmp
        size+=size_tmp
    return sum_channel, sum2_channel, size

def process(batch, path, which_file, which_batch, mean_std):
    p = batch.to_pandas()
    im = np.array(p.iloc[:,0].tolist()).reshape((-1,125,125,8))
    meta = np.array(p.iloc[:,1]).astype(np.bool_)
    return saver(path, im, meta, which_file, which_batch, mean_std)

def png_helper(im, mean_channels, std_channels, use_mean_std):
    B, C, H, W = im.shape
    sum_channel, sum2_channel, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0
    
    im[im < 1.e-3] = 0

    #Normalize
    for c in range(8):
      if use_mean_std:
        mean, std = mean_channels[c], std_channels[c]
      else:
        sum_channel[c], sum2_channel[c] = np.sum(im[:,:,:,c]), np.sum(im[:,:,:,c]**2)
        mean, std = sum_channel[c]/(B*H*W), np.sqrt(sum2_channel[c]/(B*H*W) -  (sum_channel[c]/(B*H*W))**2)

      im[:,c,:,:] = (im[:,c,:,:] - mean)/std

      #Scale to [0,255] and save as png
      max_ = im[:,:,:,c].max(axis=(1,2), keepdims=True)
      max_[max_==0] = 1
      im[:,:,:,c] *= 255/max_
      batch_std = im[:,:,:,c].std(axis=(1,2), keepdims=True)
      #Clip values < 0 or > 500 sigmas
      im[:,:,:,c][(im[:,:,:,c]<0)|(im[:,:,:,c]>500*batch_std)] = 0

    im = im.astype(np.uint8)
    
    return im, sum_channel, sum2_channel, B*H*W

def mk_png(path, img, meta, which_file, which_batch, which_pic):
    impath = path+f"/{meta}/{which_file}_{which_batch}_{which_pic}.png"
    img = np.append(img, np.ones((125,125,1)), axis=2).reshape(-1,125,3)[:,:,::-1]
    cv2.imwrite(impath, img)

def saver(path, im, meta, which_file, which_batch, mean_std):

    if not (mean_std[0] is None):
      use_mean_std = True
      mean, std = mean_std
    else:
      use_mean_std = False
      mean, std = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0])
    im, sum_channel, sum2_channel, size = png_helper(im, mean, std, use_mean_std)
   
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(mk_png, [(path, im[which_pic,:,:,:], int(meta[which_pic]), which_file, which_batch, which_pic) for which_pic in range(meta.shape[0])])

    return sum_channel, sum2_channel, size

def runner(source, indexes, target, mean_std=(None,None)):

    sumw, sumw2, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0

    files = [source+f'/BoostedTop_x1_fixed_{i}.snappy.parquet' for i in indexes]

    print("The following files will be processed")
    print(files)

    for i in tqdm(range(len(files))):
        try:
            sum_tmp, sum2_tmp, size_tmp = generate(pq.ParquetFile(files[i]), target, which_file=indexes[i], mean_std=mean_std)
  
            sumw+=sum_tmp
            sumw2+=sum2_tmp 
            size+=size_tmp
        except:
            continue

    mean, std = sumw/size, np.sqrt( sumw2/size - (sumw/size)**2 )

    print('mean and std are ', mean, std)
    print("The files were successfully generated")
    return mean, std

if __name__=='__main__':

    os.makedirs("data/top/train/0", exist_ok=True)
    os.makedirs("data/top/train/1", exist_ok=True)
    os.makedirs("data/top/test/0", exist_ok=True)
    os.makedirs("data/top/test/1", exist_ok=True)

    mean, std  = runner(source="data/top/parquet", indexes=range(1002), target="data/top/train")
   
    runner(source="data/top/parquet", indexes=range(1003,1250), target="data/top/test", mean_std=(mean, std))
