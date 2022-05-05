import torch
from torch.utils.data import Dataset
import numpy as np
import os
from os.path import isfile, join
from skimage import data, filters
import csv
import json
import nibabel as nibs
import pydicom as dcm


class UnetDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, min_probability=None):# /home/sci/hdai/Projects/LymphNodes
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.field_list = ['Series UID', 'Collection', '3rd Party Analysis', 
                      'Data Description URI', 'Subject ID', 'Study UID', 
                      'Study Description', 'Study Date', 'Series Description', 
                      'Manufacturer', 'Modality', 'SOP Class Name', 
                      'SOP Class UID', 'Number of Images', 'File Size', 
                      'File Location', 'Download Timestamp']
        self.case_info = self.get_metadata(f"{self.root_dir}/metadata.csv")
        self.bp_and_cubes_data = self.get_cubes_data()
        
        if min_probability is not None:
            self.probability_map = self.get_probability_map()
            self.probability_map /= np.sum(self.probability_map)
            prob_map_shape = self.probability_map.shape
            self.cube_extents = []
            for case, data in self.bp_and_cubes_data.items():
                case_bp_shape = data["bp_shape"]
                case_bp_extent = data["bp_extents"]
                trans_mult = [prob_map_shape[axis] / case_bp_shape[axis] for axis in range(3)]
                for cube_extent in data["cube_extents"]:
                    prob_coords = [[int(trans_mult[axis]*(cube_extent[axis][0]-case_bp_extent[axis][0])),
                                    int(trans_mult[axis]*(cube_extent[axis][1]-case_bp_extent[axis][0]))] for axis in range(3)]
                    cube_prob = np.sum(self.probability_map[prob_coords[0][0]:prob_coords[0][1],
                                                            prob_coords[1][0]:prob_coords[1][1],
                                                            prob_coords[2][0]:prob_coords[2][1]])
                    if cube_prob >= min_probability:
                        self.cube_extents.append([case, cube_extent])
                    
        else:
            self.cube_extents = []
            for case, data in self.bp_and_cubes_data.items():
                self.cube_extents.extend([[case, d] for d in data["cube_extents"]])           
        
        
    def get_probability_map(self):
        with open(f"{self.root_dir}/prob_map.npy", "rb") as f:
            p_map = np.load(f)
        return p_map
    
    def get_cubes_data(self):
        with open(f"{self.root_dir}/bp_and_cubes_data_{self.patch_size}.json", "r") as f:
            data = json.loads(f.read())
        return data
                    
    def get_metadata(self, filename):
        cases = {}
        with open(filename, 'r') as csvf:
            csv_reader = csv.reader(csvf)
            header = next(csv_reader)
            for row in csv_reader:
                if not row[4].startswith("MED"):
                    continue
                cases[row[4]] = {header[i]:row[i] for i in range(len(row))}
        return cases
                    
    def read_dicom(self, case: str):
        filename = self.root_dir + '/' + self.case_info[case]['File Location'][1:].replace('\\', '/')
        dcms = os.listdir(filename)
        dcms.sort()
        first_image = dcm.read_file(f"{filename}/{dcms[0]}")
        first_pixs = first_image.pixel_array
        volume = np.empty((first_pixs.shape[0], first_pixs.shape[1], len(dcms)))
        for idx, im in enumerate(dcms):
            pixels = dcm.read_file(f"{filename}/{im}").pixel_array
            volume[:,:,idx] = pixels.transpose()
        return volume

    def read_mask(self, case: str):
        case_name = self.case_info[case]['File Location'][1:].split('\\')[2]
        filename = f"{self.root_dir}/MED_ABD_LYMPH_MASKS/{case_name}/{case_name}_mask.nii.gz"
        mask = nibs.load(filename).get_fdata()
        
        return mask
        
    def __len__(self):
        return len(self.cube_extents)

    def get_whole_volume_and_mask(self, idx):
        case_names = list(self.case_info.keys())
        case_name = case_names[idx]
        img = torch.from_numpy(self.read_dicom(case_name))
        mask = torch.from_numpy(self.read_mask(case_name))

        sample = {
            'name': case_name,
            'img': img,
            'mask': mask
        }

        return sample

    def __getitem__(self, idx):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        case_name = self.cube_extents[idx][0]
        img = torch.from_numpy(self.read_dicom(case_name))
        
        mask = torch.from_numpy(self.read_mask(case_name))
        mask[mask>1] = 1
                
        cube_extent = self.cube_extents[idx][1]
        
        im_cube = img[cube_extent[0][0]:cube_extent[0][1],
                      cube_extent[1][0]:cube_extent[1][1],
                      cube_extent[2][0]:cube_extent[2][1]]
        im_cube[im_cube<100-300] = 100 - 300
        im_cube[im_cube>100+300] = 100 + 300

        im_cube -= torch.min(im_cube)
        im_cube /= torch.max(im_cube)
        
#         im_cube.to(device)
        
        mask_cube = mask[cube_extent[0][0]:cube_extent[0][1],
                         cube_extent[1][0]:cube_extent[1][1],
                         cube_extent[2][0]:cube_extent[2][1]]
        
#         mask_cube.to(device)

        sample = {"name": case_name,
                  "img": im_cube.unsqueeze(0),
                  "mask": mask_cube.unsqueeze(0)}
        
        return sample


class FnetDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):# /home/sci/hdai/Projects/LymphNodes
#         construct case_info dict
        self.case_info = []
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.field_list = ['Series UID', 'Collection', '3rd Party Analysis', 
                      'Data Description URI', 'Subject ID', 'Study UID', 
                      'Study Description', 'Study Date', 'Series Description', 
                      'Manufacturer', 'Modality', 'SOP Class Name', 
                      'SOP Class UID', 'Number of Images', 'File Size', 
                      'File Location', 'Download Timestamp']
        with open(f'{root_dir}/metadata.csv', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                self.case_info.append({self.field_list[i]:row[i] for i in range(len(row))})
#                 only use mediastinal lymph node
        self.case_info = self.case_info[87:]
        
    def __len__(self):
        return len(self.case_info)
        
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         construct 3d CT from dicom folder
        # '/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        relative_ct_folder_path = self.case_info[idx]['File Location'][1:].replace('\\','/')
        # '/home/sci/hdai/Projects/LymphNodes/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        ct_folder_path = f'{self.root_dir}{relative_ct_folder_path}'
        slice_name_list = [f for f in os.listdir(ct_folder_path)]
        slice_name_list.sort()
        slice_list = []
        for slice_name in slice_name_list:
            ds = dcm.dcmread(f'{ct_folder_path}/{slice_name}')
            slice_list.append(torch.from_numpy(ds.pixel_array.transpose()))
        img = torch.stack(slice_list,-1).to(device)
        
#         load 3d mask
        case_name = self.case_info[idx]['File Location'][17:30].replace('\\','/')
        mask_path = f'{self.root_dir}/MED_ABD_LYMPH_MASKS/{case_name}/{case_name}_mask.nii.gz'
        mask = torch.from_numpy(nibs.load(mask_path).get_fdata()).to(device)
        mask[mask>1] = 1
        
        half_patch_size = int(self.patch_size/2)
        idx_x, idx_y, idx_z = torch.where(mask!=0)
        centroid_x, centroid_y, centroid_z = 256, 256, 300
        if int(torch.mean(idx_x.float())) < mask.shape[0]-half_patch_size and int(torch.mean(idx_x.float())) > half_patch_size:
            centroid_x = int(torch.mean(idx_x.float()))
        if int(torch.mean(idx_y.float())) < mask.shape[1]-half_patch_size and int(torch.mean(idx_y.float())) > half_patch_size:
            centroid_y = int(torch.mean(idx_y.float()))
        if int(torch.mean(idx_z.float())) < mask.shape[2]-half_patch_size and int(torch.mean(idx_z.float())) > half_patch_size:
            centroid_z = int(torch.mean(idx_z.float()))
        
        image_list, mask_list = [], []
        
        mask = mask[centroid_x-half_patch_size:centroid_x+half_patch_size, \
                    centroid_y-half_patch_size:centroid_y+half_patch_size, \
                    centroid_z-half_patch_size:centroid_z+half_patch_size]
        
        img[img<70-750]=70-750
        img[img>70+750]=70+750
        img = img - torch.min(img)
        img = img/(torch.max(img)-torch.min(img))
        
        for i in range(4):
            image_list.append(img[centroid_x-int(half_patch_size/2**i):centroid_x+int(half_patch_size/2**i), \
                                  centroid_y-int(half_patch_size/2**i):centroid_y+int(half_patch_size/2**i), \
                                  centroid_z-int(half_patch_size/2**i):centroid_z+int(half_patch_size/2**i)])
#             mask_list.append(mask[centroid_x-int(half_patch_size/2**i):centroid_x+int(half_patch_size/2**i), \
#                                   centroid_y-int(half_patch_size/2**i):centroid_y+int(half_patch_size/2**i), \
#                                   centroid_z-int(half_patch_size/2**i):centroid_z+int(half_patch_size/2**i)])

        sample = {  'name' : case_name,
                    'img0' : image_list[0].unsqueeze(0),
                    'img1' : image_list[1].unsqueeze(0),
                    'img2' : image_list[2].unsqueeze(0),
                    'img3' : image_list[3].unsqueeze(0),
                    'mask' : mask.long().unsqueeze(0)}
#                     'mask' : torch.stack((mask.long(),1-mask.long()),0)}
        
        return sample