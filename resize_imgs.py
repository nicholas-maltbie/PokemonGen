import pandas
from multiprocessing import Pool
import argparse
from PIL import Image
import os

def scale_img(input_path, output_path, width=256, height=256, buffer=0.1, background="white", log=True):
    img = Image.open(input_path)
    
    (cur_width, cur_height) = img.size
    
    deform = min(width * (1.0 - buffer) / cur_width, height * (1.0 - buffer) / cur_height)
    if(log):
        print("Scaling", input_path, "to", output_path, "by factor of", round(deform, 3), "with padding")
    desired_size = (int(cur_width * deform), int(cur_height * deform))
    delta_x = (width - desired_size[0])
    delta_y = (height - desired_size[1])
    im_new = Image.new("RGB", (width, height), color=background)
    im_new.paste(img.resize(desired_size, Image.LANCZOS), (delta_x // 2, delta_y // 2))
    im_new.save(output_path)

def main(image_dir, data_file, output_dir, width=256, height=256, num_tasks = 32, log=False, buffer=0.1):
    img_data = pandas.read_csv(data_file, header=None)
    
    img_list = img_data.iloc[:,1]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pool = Pool(num_tasks)
    
    for img in img_list:
        pool.apply_async(scale_img, (os.path.join(image_dir, img) + ".jpg", os.path.join(output_dir, img) + ".jpg", width, height, buffer, 'white', log))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and categorize images")
    parser.add_argument('image_dir', help="Directory containing all images to be scaled")
    parser.add_argument('data_file', help="File with groupings for data")
    parser.add_argument('output_dir', help="Directory to write output images to")
    parser.add_argument('-s', metavar="256", type=int, help="Size to scale image to (default 256)", default=256)
    parser.add_argument('-b', metavar="0.1", type=float, help="Percent of image to use as buffer (default 0.1)", default=0.1)

    args = parser.parse_args()
    main(args.image_dir, args.data_file, args.output_dir, width=args.s, height=args.s, buffer=args.b)

