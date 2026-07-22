# Packages
import apoc
import os
import skimage as ski
import sklearn as skl
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
import numpy as np
import napari


# Image
path = r'C:/Users/Tibo/Documents/Internship_LCB_Myxococcus_xanthus/Images/Microscopy/Raw/Output_pc/PC_20260608_isolated_spores_plate_1_Heat_003.nd2.tif'
img = ski.io.imread(path)


# Take a look with a max projection
projection= img.mean(axis=0)
ski.io.imshow(projection)


# Annotate
annotation = np.zeros(projection.shape)
annotation[2100:2150,2400:3200] = 1
annotation[45:500,10:100] = 2
ski.io.imshow(annotation, vmin=0, vmax=2)


# Generating a feature stack
def generate_feature_stack(image):
    # determine features
    blurred = ski.filters.gaussian(projection, sigma=2)
    edges = ski.filters.sobel(blurred)

    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values in a 1-D format here. 
    feature_stack = [
        image.ravel(),
        blurred.ravel(),
        edges.ravel()
    ]
    
    # return stack as numpy-array
    return np.asarray(feature_stack)

feature_stack = generate_feature_stack(projection)

# show feature images
fig, axes = plt.subplots(1, 3, figsize=(10,10))

# reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
axes[0].imshow(feature_stack[0].reshape(projection.shape), cmap=plt.cm.gray)
axes[1].imshow(feature_stack[1].reshape(projection.shape), cmap=plt.cm.gray)
axes[2].imshow(feature_stack[2].reshape(projection.shape), cmap=plt.cm.gray) 


# Formatting data
def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = feature_stack.T
    # make the annotation 1-dimensional
    y = annotation.ravel()
    
    # remove all pixels from the feature and annotations which have not been annotated
    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y

X, y = format_data(feature_stack, annotation)

print("input shape", X.shape)
print("annotation shape", y.shape)



# Training the random forest classifier
classifier = skl.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X, y)


# Predicting pixel classes
res = classifier.predict(feature_stack.T) - 1 # we subtract 1 to make background = 0
ski.io.imshow(res.reshape(projection.shape))


# Interactive segmentation
# start napari
viewer = napari.Viewer()

# add image
viewer.add_image(img)

# add an empty labels layer and keet it in a variable
labels = viewer.add_labels(np.zeros(img.shape).astype(int))

manual_annotations = labels.data

ski.io.imshow(manual_annotations, vmin=0, vmax=2)



# generate features (that's actually not necessary, 
# as the variable is still there and the image is the same. 
# but we do it for completeness)
feature_stack = generate_feature_stack(img)
X, y = format_data(feature_stack, manual_annotations)

# train classifier
classifier = skl.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X, y)

# process the whole image and show result
result_1d = classifier.predict(feature_stack.T)
result_2d = result_1d.reshape(img.shape)
ski.io.imshow(result_2d)


viewer.add_labels(result_2d)
napari.utils.nbscreenshot(viewer)



