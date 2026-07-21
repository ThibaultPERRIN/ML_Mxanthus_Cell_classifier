# Packages
import os
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import tifffile
import pathlib
import imageio.v3 as iio

# # Import image and take look to parameters
# path = r'C:/Users/nmassoulie/Desktop/input_PC_images/Segmented/Image.tif'

# print('Path: ', os.path.exists(path))

# img = ski.io.imread(path)
# #print(img)
# img.shape


# %% Functions

# Function separation spore-vc arrays
def arrays_separation(img):

    """
    ## Copy image into two seperate class of arrays (spores and vc)
    ## Condition: normalize image segmentation
    input:
        - Arrays: image
    output:
        - Arrays: spore image
        - Arrays: vc image
    """
    
    # Background correction
    correction = np.ones((17, 3200, 3200))
    img = img + correction
    img[img == 2] = 0 # Bkgrd

    # Seperation spore-VC arrays
    img_sp, img_vc = np.copy(img), np.copy(img)
    
    img_sp[img_sp > 1] = 0
    img_vc[img_vc < 2] = 0
    
    return img_sp, img_vc


# Function object classification in df
def classification(spores, vegetative, image_id):

    """
    ## Take images (spore image, vegetative image):
            - Labels and measures labelled objects (properties)
            - For each image: Creates a mask, computes the volume, surface area and sphericity of each individual object
    ## Conditions:
            - Normalized segmented images
            - Two seperated sub-images (spore image, vegetative image)
    input:
        - Arrays: image (spore image)
        - Arrays: image (vegetative image)
    output:
        - Dataframe: classification objects (properties) in a table
    """
    
    img_grouped = [
        ('spore', spores), 
        ('vc', vegetative)
    ]
    spacing = (1, 0.1621455, 0.1621455)
    voxel_volume = np.prod(spacing)
    results = pd.DataFrame()
        
    for cell_type, i in img_grouped:            
        # Label objects
        labels = ski.measure.label(i)
        objects = ski.measure.regionprops(
            labels, 
            spacing = spacing
        )
            
        # Properties
        for prop in objects:

            # Objects
            label = prop.label
            mask = labels == label
    
            # Volume
            volume = mask.sum() * voxel_volume
            
            # Surface mesh
            verts, faces, _, _ = ski.measure.marching_cubes(mask.astype(float), level=0.5, spacing=spacing)
            surface_area = ski.measure.mesh_surface_area(verts, faces)
                
            # Sphericity
            psi = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area

            # Dataframe
            data={
                "Image_ID": image_id,
                "Label": prop.label,
                "Cell_type": cell_type,
                "Area": prop.area,
                "Volume": volume,
                "Surface_area": surface_area,
                "Sphericity": psi
            }
            d = pd.DataFrame(data, index=[0])
            results = pd.concat([results, d], ignore_index = True)
    
    return results


# %% Script image processing
# Read image folder
# images = []
# for file in pathlib.Path(r'C:/Users/nmassoulie/Desktop/input_PC_images/Segmented/').iterdir():
#     if not file.is_file():
#         continue
    
#     file_name, file_extension = os.path.splitext(file.name)
#     images.append((file_name, iio.imread(file)))
    
#     for file_name, image in images:
#         img_sp, img_vc = arrays_separation(image)
#         results = classification(img_sp, img_vc, file_name)


all_results = pd.DataFrame()

for file in pathlib.Path(r'C:/Users/nmassoulie/Desktop/input_PC_images/Segmented').iterdir():
    if not file.is_file():
        continue

    file_name = file.stem          
    image = iio.imread(file)

    img_sp, img_vc = arrays_separation(image)
    results = classification(img_sp, img_vc, file_name)

    all_results = pd.concat([all_results, results], ignore_index=True)
    all_results = all_results[(all_results['Area'] > 5)]
    

        
        