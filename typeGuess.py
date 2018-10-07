import pandas
import os
import numpy as np
import argparse
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

types = np.array([elem.upper() for elem in ["Normal", "Fire", "Fighting", "Water", "Flying", "Grass", "Poison", "Electric", "Ground", "Psychic", "Rock", "Ice", "Bug", "Dragon", "Ghost", "Dark", "Steel", "Fairy"]])
num_types = len(types)
types_idx = {types[idx]:idx for idx in range(num_types)}

def read_file(file):
    input_data = pandas.read_csv(file, index_col=0, header=None, keep_default_na=False)

    names = input_data.iloc[:,0]
    types = np.apply_along_axis(get_vector_from_types, 1, input_data.iloc[:,[1,2]])
    
    return names, types

def get_vector_from_types(types):
    return sum([np.eye(num_types)[types_idx[type.upper()]] for type in types if type != ""])

def get_types_from_vector(vec):
    return types[vec]

def read_imgs(img_list, image_dir):
    images = []
    for img in img_list:
        if os.path.exists(os.path.join(image_dir, img) + ".jpg"):
            images.append(img_to_array(load_img(os.path.join(image_dir, img) + ".jpg")))
    return np.array(images)

def build_model(args):
    model = Sequential()
    
    for cnn_layer in range(args.convolutions):
        model.add(Conv2D(args.hidden_size, (args.filter_size, args.filter_size), input_shape=(args.image_width, args.image_height, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(args.pool_size, args.pool_size)))

    model.add(Flatten())
    model.add(Dense(args.dense_size))
    model.add(Activation('relu'))
    model.add(Dropout(args.dropout_rate))
    model.add(Dense(num_types))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def main(args):
    train_names, tain_lab = read_file(args.train_file)
    val_names, val_lab = read_file(args.val_file)
    test_names, test_lab = read_file(args.test_file)
    
    train_img = read_imgs(train_names, args.image_dir)
    val_img = read_imgs(val_names, args.image_dir)
    test_img = read_imgs(test_names, args.image_dir)
    
    datagen = ImageDataGenerator(horizontal_flip=True, 
        rotation_range = 20, 
        featurewise_center=True, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        rescale=1.0/255)

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    datagen.fit(train_img)

    train_gen = datagen.flow(np.array(train_img), y=tain_lab, batch_size=args.batch_size, shuffle=True)

    val_gen = test_datagen.flow(np.array(val_img), y=val_lab, batch_size=args.batch_size, shuffle=True)
    
    model = build_model(args)
    
    # Create log for tensorboard
    tbCallBack = TensorBoard(log_dir=args.log_dir, histogram_freq=0, write_graph=True, write_images=True)

    # Create callback for saving
    checkpointCallback = ModelCheckpoint(os.path.splitext(args.model_name)[0] + "-{epoch:03d}" + os.path.splitext(args.model_name)[1], period=5)
    
    # Fit the model to the data
    model.fit_generator(train_gen,
            steps_per_epoch=args.train_steps,
            epochs=args.num_epochs,
            validation_data=val_gen,
            validation_steps=args.train_steps,
            callbacks=[tbCallBack, checkpointCallback])

    # Save model after training
    model.save(args.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train generator to guess types of pokemon")
    parser.add_argument('image_dir', help="directory with all images")
    parser.add_argument('train_file', help="File with list and labels of training files")
    parser.add_argument('val_file', help="File with list and labels of validation files")
    parser.add_argument('test_file', help="File with list and labels of test files")
    parser.add_argument('model_name', help="Name to save model to")
    parser.add_argument('log_dir', help="Log to write training data to")
    
    parser.add_argument('-nfil', dest="hidden_size", metavar="32", help="Number of filters to use (default 32)", type=int, default=32)
    parser.add_argument('-sfil', dest="filter_size", metavar="4", help="Size of filter to use (in pixels) (default 4)", type=int, default=4)
    parser.add_argument('-p', dest="pool_size", metavar="2", help="Size of max pool for making layers smaller (default 2)", type=int, default=2)
    parser.add_argument('-conv', dest="convolutions", help="Layers of convolutions to add", type=int, default=3)
    parser.add_argument('-b', dest="batch_size", metavar="16", help="Batch size to use when training network (default 16)", type=int, default=16)
    parser.add_argument('-dense', dest="dense_size", metavar="32", help="Size of dense layer (default 32)", type=int, default=32)
    parser.add_argument('-dropout', dest="dropout_rate", metavar="0.2", help="Dropout rate when reading hidden layer (default 0.2)", type=float, default=0.2)
    parser.add_argument('-t', dest="train_steps", metavar="64", help="Number of training steps per epoch (default 64)", type=int, default=64)
    parser.add_argument('-e', dest="num_epochs", metavar="256", help="Number of epochs to complete (default 256)", type=int, default=256)
    parser.add_argument('-width', dest="image_width", metavar="256", help="width of images in pixels (default 256)", type=int, default=256)
    parser.add_argument('-height', dest="image_height", metavar="256", help="height of images in pixels (default 256)", type=int, default=256)

    args = parser.parse_args()
    main(args)
