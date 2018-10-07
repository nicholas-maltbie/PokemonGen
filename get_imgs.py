from multiprocessing import Pool
import argparse
import urllib.request
import re
import os

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

def download_img(name, url, output_file, log=True):
    if not os.path.isfile(output_file):
        if log:
            print("Downloading", name, "from", url, "to", output_file)
        
        request = urllib.request.Request(url, headers=hdr)
        response = urllib.request.urlopen(request)

        data = response.read()

        with open(output_file, 'wb') as img:
            img.write(data)
     
def main(image_dir = "imgs", pokemon_list = "list.csv", num_tasks = 32, log=True):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    pool = Pool(num_tasks)
    for line in open(pokemon_list):
        num, name, type1, type2 = line.strip().split(',')
        name_no_space = name.replace(" ", "-").replace(".", "").lower()

        url = "https://img.pokemondb.net/artwork/large/" + name_no_space + ".jpg"
        output_file = os.path.join(image_dir, name) + ".jpg"
        
        pool.apply_async(download_img, (name, url, output_file, log))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downlaod all pokemon images")
    parser.add_argument("download_dir", help="Directory to downlaod images to")
    parser.add_argument("pokemon_list", help="List of all pokemon, numbers and related data")
    parser.add_argument("-q", "--quiet", dest="quiet", action='store_true', help="Should downlaod information be shwon")
    parser.add_argument("-t", "--num_threads", dest="num_threads", metavar=32, type=int, help="Number of threads to use when downloading images (default 32)")
    args = parser.parse_args()
    
    main(image_dir=args.download_dir, pokemon_list=args.pokemon_list, num_tasks=args.num_threads, log=not args.quiet)
    
