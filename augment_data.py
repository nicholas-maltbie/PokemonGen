import pandas
import shutil
import argparse
from PIL import Image, ImageDraw
from multiprocessing import Pool
import os
import numpy as np
import math


class augmentor:
    def __init__(self, image_dir, data_file, output_dir, num_create=100000, log=True, num_tasks=64, rot_range=40, shift_range=0.2, zoom_range=0.1, background="white"):
        self.image_dir = image_dir
        self.data_file = data_file
        self.output_dir = output_dir
        self.num_create = num_create
        self.log = log
        self.num_tasks = num_tasks
        self.rot_range = rot_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.background = background
    
    def copy_img(self, src_file, dest_file):
        if self.log:
            print("Copying", src_file, "to", dest_file)
        shutil.copy(src_file, dest_file)
    
    def generate_images(self, num_gen, thread=0):
        generator = self.datagen.flow(np.array(self.images), batch_size=self.batch_size, shuffle=False, save_to_dir=self.output_dir, save_format="jpg")
        
        for batch in range(math.ceil(num_gen / self.batch_size)):
            img = next(generator)
            idx = (generator.batch_index - 1) * generator.batch_size
            if idx < 0:
                idx = len(self.img_list) + idx
            subset = list()
            if idx + generator.batch_size > len(self.img_list):
                subset = np.array(self.img_list[idx:] + self.img_list[:len(self.img_list) - idx])
            else:
                subset = np.array(self.img_list[idx : idx + generator.batch_size])
            if self.log:
                print("[thread:" + str(thread) + "] generating batch", batch, "from images: ",
                str(subset))
    
    def augment_file(self, image, save_path):
        dup = image.copy().convert("RGBA")
        resize_fac = 1 - (np.random.rand() * self.zoom_range * 2 - self.zoom_range)
        rot_deg = np.random.rand() * self.rot_range * 2 - self.zoom_range
        shift_x = round(np.random.rand() * self.shift_range * image.width * 2 - self.shift_range * image.width)
        shift_y = round(np.random.rand() * self.shift_range * image.height * 2 - self.shift_range * image.height)
        flip = np.random.rand() > 0.5
        
        if flip:
            dup = dup.transpose(Image.FLIP_LEFT_RIGHT)
        dup = dup.resize((round(image.width * resize_fac), round(image.height * resize_fac)), Image.LANCZOS)
        dup = dup.rotate(rot_deg, resample=Image.BICUBIC, expand=1)
        tmp = Image.new("RGBA", dup.size, self.background)
        dup = Image.composite(dup, tmp, dup)
        
        deltax = dup.width - image.width
        deltay = dup.height - image.height
        
        im_new = Image.new("RGBA", (image.width, image.height), color=self.background)
        im_new.paste(dup, (shift_x - deltax // 2, shift_y - deltay // 2))
        if self.log:
            print("Generating file", save_path)
        im_new.convert("RGB").save(save_path)
    
    def augment_files(self):
        img_data = pandas.read_csv(self.data_file, header=None)

        self.img_list = img_data.iloc[:,1]
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
             
        pool = Pool(self.num_tasks)
        
        augmentations_per = math.ceil(self.num_create / len(self.img_list))
        
        for img in self.img_list:
            src_file = os.path.join(self.image_dir, img) + ".jpg"
            dest_file = os.path.join(self.output_dir, img) + ".jpg"
            if os.path.exists(src_file):
                img_dat = Image.open(src_file)
                pool.apply_async(self.copy_img, (src_file, dest_file))
                for iter in range(augmentations_per):
                    pool.apply_async(self.augment_file, (img_dat, os.path.join(self.output_dir, img) + "-" + str(iter) + ".jpg"))
        
        pool.close()
        pool.join()
        print("Done generating images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and categorize images")
    parser.add_argument('image_dir', help="Directory containing all images to be scaled")
    parser.add_argument('data_file', help="File with groupings for data")
    parser.add_argument('output_dir', help="Directory to write output images to")
    parser.add_argument('-n', '--num_images', dest="num_create", metavar="100000", type=int, help="Number of images to generate (default 100000)", default=100000)
    parser.add_argument('-r', '--rotation_range', dest="rot_range", metavar="40", type=int, help="Range of degrees that the images can be rotated -- plus or minus (default 40)", default=40)
    parser.add_argument('-s', '--shift_range', dest="shift_range", metavar="0.2", type=float, help="Percent that an image can be shifed in the horizontal and vertical axis (default 0.2)", default=0.2)
    parser.add_argument('-z', '--zoom_range', dest="zoom_range", metavar="0.1", type=float, help="The range in which an image can be zoomed in and out (default 0.1)", default=0.1)
    parser.add_argument("-q", "--quiet", dest="quiet", action='store_true', help="Should downlaod information be shwon")
    parser.add_argument("-t", "--num_threads", dest="num_threads", metavar=64, type=int, help="Number of threads to use for generating images and saving them (default 64)", default=64)
    parser.add_argument("-b", "--background", dest="background", metavar='white', help="background color of images (default white)", default='white')

    args = parser.parse_args()
    file_aug = augmentor(args.image_dir, args.data_file, args.output_dir, num_create=args.num_create, rot_range = args.rot_range, shift_range=args.shift_range, zoom_range = args.zoom_range, log= not args.quiet, num_tasks=args.num_threads, background=args.background)
    file_aug.augment_files()
