from fastai.vision import *

# Define path to the image folder
path = Path('path/to/image/folder')

# Create a data bunch
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

# Create a learner
learn = cnn_learner(data, models.resnet50, metrics=accuracy)

# Fit the model
learn.fit_one_cycle(4)

# Save the model
learn.save('weed-detection-resnet50')

# Predict on new images
img = open_image(path/'new_image.jpg')
pred_class, pred_idx, outputs = learn.predict(img)

# Print the prediction
