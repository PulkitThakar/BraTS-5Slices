import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

original_data = os.listdir("D:/Pulkit/Cool Stuff/Work/DL/BraTS/Dataset/2018 Dataset/Training/HGG")
original_data_path = "D:/Pulkit/Cool Stuff/Work/DL/BraTS/Dataset/2018 Dataset/Training/HGG"
target_seg = "D:/Pulkit/Cool Stuff/Work/DL/UNet/U-NET/Data 5 slices/seg"
target_flair = "D:/Pulkit/Cool Stuff/Work/DL/UNet/U-NET/Data 5 slices/flair"
target_t1 = "D:/Pulkit/Cool Stuff/Work/DL/UNet/U-NET/Data 5 slices/t1"
target_t1ce = "D:/Pulkit/Cool Stuff/Work/DL/UNet/U-NET/Data 5 slices/t1ce"
target_t2 = "D:/Pulkit/Cool Stuff/Work/DL/UNet/U-NET/Data 5 slices/t2"

count = 1

for i in original_data:

    temp = nib.load(os.path.join(original_data_path, i, i + "_seg.nii.gz"))
    temp = np.array(temp.dataobj)

    temp_1 = np.empty((temp.shape[0], temp.shape[1], 5))

    my_list = []
    for ii in range(temp.shape[2]):
        temp_val = np.count_nonzero(temp[:, :, ii])
        my_list.append(temp_val)

    #print(len(my_list))
    my_list_1 = []

    for ii in my_list:
        my_list_1.append(ii)

    my_list_1.sort(reverse = True)

    #print(my_list[0])
    #print("Hi")

    my_list_1_indices = []
    for ii in range(5):
    
        if (ii >= 1):
            
            if my_list_1[ii] == my_list_1[ii-1]:
                temp_ind = my_list_1_indices[ii-1]
                temp_ind += 1
                temp_ind_1 = my_list[temp_ind:].index(my_list_1[ii])
                my_list_1_indices.append(temp_ind + temp_ind_1)
                del temp_ind, temp_ind_1
                
            else:
                my_list_1_indices.append(my_list.index(my_list_1[ii]))
        else:
            my_list_1_indices.append(my_list.index(my_list_1[ii]))


    temp = nib.load(os.path.join(original_data_path, i, i + "_flair.nii.gz"))
    aff = np.array(temp.affine)
    temp = np.array(temp.dataobj)

    for ii in range(temp_1.shape[2]):
        temp_1[:, :, ii] = temp[:, :, my_list_1_indices[ii]]

    img = nib.Nifti1Image(temp_1, aff)
    nib.save(img, os.path.join(target_flair, i + "_flair.nii.gz"))

    
    
    temp = nib.load(os.path.join(original_data_path, i, i + "_seg.nii.gz"))
    aff = np.array(temp.affine)
    temp = np.array(temp.dataobj)

    for ii in range(temp_1.shape[2]):
        temp_1[:, :, ii] = temp[:, :, my_list_1_indices[ii]]

    img = nib.Nifti1Image(temp_1, aff)
    nib.save(img, os.path.join(target_seg, i + "_seg.nii.gz"))


    temp = nib.load(os.path.join(original_data_path, i, i + "_t1.nii.gz"))
    aff = np.array(temp.affine)
    temp = np.array(temp.dataobj)

    for ii in range(temp_1.shape[2]):
        temp_1[:, :, ii] = temp[:, :, my_list_1_indices[ii]]

    img = nib.Nifti1Image(temp_1, aff)
    nib.save(img, os.path.join(target_t1, i + "_t1.nii.gz"))


    temp = nib.load(os.path.join(original_data_path, i, i + "_t1ce.nii.gz"))
    aff = np.array(temp.affine)
    temp = np.array(temp.dataobj)

    for ii in range(temp_1.shape[2]):
        temp_1[:, :, ii] = temp[:, :, my_list_1_indices[ii]]

    img = nib.Nifti1Image(temp_1, aff)
    nib.save(img, os.path.join(target_t1ce, i + "_t1ce.nii.gz"))


    temp = nib.load(os.path.join(original_data_path, i, i + "_t2.nii.gz"))
    aff = np.array(temp.affine)
    temp = np.array(temp.dataobj)

    for ii in range(temp_1.shape[2]):
        temp_1[:, :, ii] = temp[:, :, my_list_1_indices[ii]]

    img = nib.Nifti1Image(temp_1, aff)
    nib.save(img, os.path.join(target_t2, i + "_t2.nii.gz"))