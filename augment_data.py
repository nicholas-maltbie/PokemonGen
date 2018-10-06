import pandas
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import os
import numpy as np
import math

def main(image_dir, data_file, output_dir, num_create=100000, log=True):
    img_data = pandas.read_csv(data_file, header=None)

    img_list = img_data.iloc[:,1]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    datagen = ImageDataGenerator(horizontal_flip=True, 
        rotation_range = 20, 
        featurewise_center=True, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1)
    
    
    images = []
    
    for img in img_list:
        if os.path.exists(os.path.join(image_dir, img) + ".jpg"):
            images.append(img_to_array(load_img(os.path.join(image_dir, img) + ".jpg")))
    
    datagen.fit(images)
    
    batch_size = 128
    generator = datagen.flow(np.array(images), batch_size=batch_size, shuffle=False, save_to_dir=output_dir, save_format="jpg")
    
    for _ in range(math.ceil(num_create / batch_size)):
        img  = next(generator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and categorize images")
    parser.add_argument('image_dir', help="Directory containing all images to be scaled")
    parser.add_argument('data_file', help="File with groupings for data")
    parser.add_argument('output_dir', help="Directory to write output images to")
    parser.add_argument('-n', metavar="100000", type=int, help="Number of images to generate (default 100000)", default=100000)

    args = parser.parse_args()
    main(args.image_dir, args.data_file, args.output_dir)

