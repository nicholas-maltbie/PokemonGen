# Pokemon Generator

Project to attempt to use a DCGAN (Deep Convolutional Generative Adverserial Network) to generate new pokemon sprites.

Using anaconda and python3 to generate data and run DCGAN. Environment uses packages in packages.txt. Create a new environment with the following command 

```bash
conda create -n pokemon-dcgan --file packages.txt
conda activate pokemon-dcgan
```

Download images from pokemonDB.net with the get_imgs.py script then resize these images to a constant scale with padding through the resize_imgs.py script.

After all the images have been resized, they can then be augmented by rotating, scaling and translating the images

```bash
python get_imgs.py data/pokemon_imgs list.csv
python resize_imgs.py data/pokemon_imgs list.csv data/pokemon_imgs_128 -s 128 -b 0.25
python augment_data.py data/pokemon_imgs_128 list.csv data/pokemon_imgs_128_aug
```

Now that we have the full dataset, this can now be used to generate the DCGAN for creating pokemon images. The size of the images can be reduced with the -s parameter to allow this to run faster. This uses the DCGAN git submodule. This submodule must first be initialized to ensure that this works properly.

```bash
git submodule init

python DCGAN-tensorflow/main.py --input_height=128 --output_height=128 --checkpoint_dir=data/checkpoint_128 --sample_dir=data/samples_128 --train --crop=False --visualize=True --dataset=pokemon_imgs_128_aug
```
