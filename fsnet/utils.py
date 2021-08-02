from __future__ import division
import numpy as np
import nibabel as nib
import copy
from glob import glob
from scipy.ndimage import rotate
from random import randint
from skimage.measure import compare_psnr, compare_ssim

def mae_psnr_ssim(prediction, ground_truth):

    prediction = prediction*3276.7-1024
    ground_truth = ground_truth*3276.7-1024

    mae = np.mean(np.abs(np.subtract(prediction, ground_truth)))
    psnr = compare_psnr(ground_truth,prediction,3276.7)
    ssim = compare_ssim(ground_truth,prediction,data_range=3276.7) 
    return mae,psnr,ssim


##############################
# process .nii data
def load_data_pairs(data_dir):
    """load all volume pairs"""
    
    pair_list = glob('{}/*.nii.gz'.format(data_dir))
    pair_list.sort()
    
    img_clec = []
    img_clec_aff = []
    label_clec_syn = []

    for k in range(0, len(pair_list), 2):
        img_path = pair_list[k+1]
        lab_path_syn = pair_list[k]

        vol_file = nib.load(img_path)
        img_aff = vol_file.affine
        img_data = vol_file.get_data().copy()
        
        lab_data_syn = nib.load(lab_path_syn).get_data().copy()
        
        img_data = img_data.astype('float32')
        lab_data_syn = lab_data_syn.astype('float32')
               
        lab_data_syn = lab_data_syn/255.0
        
        mean_temp = np.mean(img_data)
        dev_temp = np.std(img_data)
        img_data = (img_data - mean_temp) / dev_temp

        img_clec.append(img_data)
        label_clec_syn.append(lab_data_syn)
        img_clec_aff.append(img_aff)

    return img_clec, label_clec_syn, img_clec_aff

# reimplementate get_batch_patches by zenglin
def get_batch_patches(rand_img, rand_label_syn, patch_dim, batch_size=1, smooth_r = 8, rot_flag=True):
    """generate a batch of paired patches for training"""
    batch_img = np.zeros([batch_size, patch_dim[0], patch_dim[1], patch_dim[2], 1]).astype('float32')
    batch_label_syn = np.zeros([batch_size, patch_dim[0], patch_dim[1], patch_dim[2],1]).astype('float32')
    batch_label_high = np.zeros([batch_size, patch_dim[0], patch_dim[1], patch_dim[2],1]).astype('float32')

    rand_img = rand_img.astype('float32')
    rand_label_syn = rand_label_syn.astype('float32')
    l, w, h = rand_img.shape

    for k in range(batch_size):
        # randomly select a box anchor        
        l_rand = randint(0, l - patch_dim[0])
        w_rand = randint(0, w - patch_dim[1])
        h_rand = randint(0, h - patch_dim[2])
        
        pos = np.array([l_rand, w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(rand_img[pos[0]:pos[0]+patch_dim[0], pos[1]:pos[1]+patch_dim[1], pos[2]:pos[2]+patch_dim[2]])
        label_temp_syn = copy.deepcopy(rand_label_syn[pos[0]:pos[0]+patch_dim[0], pos[1]:pos[1]+patch_dim[1], pos[2]:pos[2]+patch_dim[2]])
        # possible augmentation
        # rotation
        if rot_flag and np.random.random() > 0.65:
            # print 'rotating patch...'
            rand_angle = [-10, 10]
            np.random.shuffle(rand_angle)
            img_norm = rotate(img_norm, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
            label_temp_syn = rotate(label_temp_syn, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)

        batch_img[k, :, :, :, 0] = img_norm
        batch_label_syn[k, :, :, :, 0] = label_temp_syn
        batch_label_high[k, :, :, :, 0] = label_temp_syn-uniform_filter(label_temp_syn,smooth_r,mode='constant')

    return batch_img, batch_label_syn, batch_label_high


# calculate the cube information
def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita    
    
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap


# decompose volume into list of cubes
def decompose_vol2cube(vol_data, cube_size, n_chn, ita):
    cube_list = []
    # get parameters for decompose
    fold, ovlap = fit_cube_param(vol_data.shape, cube_size, ita)
    dim = np.asarray(vol_data.shape)
    # decompose
    for R in range(0, fold[0]):
        r_s = R*cube_size[0] - R*ovlap[0]
        r_e = r_s + cube_size[0]
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size[0]
            r_e = r_s + cube_size[0]
        for C in range(0, fold[1]):
            c_s = C*cube_size[1] - C*ovlap[1]
            c_e = c_s + cube_size[1]
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size[1]
                c_e = c_s + cube_size[1]
            for H in range(0, fold[2]):
                h_s = H*cube_size[2] - H*ovlap[2]
                h_e = h_s + cube_size[2]
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size[2]
                    h_e = h_s + cube_size[2]
                # partition multiple channels
                cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
                cube_batch = np.zeros([1, cube_size[0], cube_size[1], cube_size[2], n_chn]).astype('float32')
                cube_batch[0, :, :, :, 0] = copy.deepcopy(cube_temp)
                # save
                cube_list.append(cube_batch)

    return cube_list

# compose list of label cubes into a label volume
def compose_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')
    flag_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('int32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size[0] - R*ovlap[0]
        r_e = r_s + cube_size[0]
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size[0]
            r_e = r_s + cube_size[0]
        for C in range(0, fold[1]):
            c_s = C*cube_size[1] - C*ovlap[1]
            c_e = c_s + cube_size[1]
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size[1]
                c_e = c_s + cube_size[1]
            for H in range(0, fold[2]):
                h_s = H*cube_size[2] - H*ovlap[2]
                h_e = h_s + cube_size[2]
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size[2]
                    h_e = h_s + cube_size[2]
                # histogram for voting
                flag_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] += 1
                # accumulation
                label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] += cube_list[p_count][0,:,:,:,:]

                p_count += 1

    flag_classes_mat[flag_classes_mat==0] += 1 
    compose_vol = label_classes_mat/flag_classes_mat

    return compose_vol
