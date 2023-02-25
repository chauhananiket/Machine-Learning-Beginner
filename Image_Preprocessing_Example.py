from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



img =image.load_img(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Breast_Cancer_Dataset\IDC_regular_ps50_idx5\8863\1\8863_idx5_x1001_y851_class1.png')  # this is a PIL image
x = image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 1
for batch in datagen.flow(x, batch_size=1, save_to_dir=r'D:\Downloads\Image_Preprocessing_Example', save_prefix='8863_idx5_x1001_y851_class1_'+str(i), save_format='png'):
    i += 1
    if i >20:
        break  # otherwise the generator would loop indefinitely
        