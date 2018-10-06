import sys
import argparse
import pandas
import numpy as np
import re
from sklearn.model_selection import train_test_split

types = np.array([elem.upper() for elem in ["Normal", "Fire", "Fighting", "Water", "Flying", "Grass", "Poison", "Electric", "Ground", "Psychic", "Rock", "Ice", "Bug", "Dragon", "Ghost", "Dark", "Steel", "Fairy"]])
num_types = len(types)
types_idx = {types[idx]:idx for idx in range(num_types)}

def get_vector_from_types(types):
    return sum([np.eye(num_types)[types_idx[type.upper()]] for type in types if type != ""])

def get_types_from_vector(vec):
    return types[vec]

def main(input_file, output_train, output_validation, output_test):
    input_data = pandas.read_csv(input_file, index_col=0, header=None, keep_default_na=False)

    indices = np.arange(input_data.shape[0])
    types = np.apply_along_axis(get_vector_from_types, 1, input_data.iloc[:,[1,2]])
    
    train_idx, other_idx, _, other_lab = train_test_split(indices, types, test_size=0.33)
    
    val_idx, test_idx, _, _ = train_test_split(other_idx, other_lab, test_size=0.5)
    
    train_data = pandas.DataFrame(input_data.iloc[train_idx,:])
    val_data = pandas.DataFrame(input_data.iloc[val_idx,:])
    test_data = pandas.DataFrame(input_data.iloc[test_idx,:])
    
    train_data.to_csv(output_train, header=None)
    val_data.to_csv(output_validation, header=None)
    test_data.to_csv(output_test, header=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data csv into train, validation and test sets based on pokemon type")
    parser.add_argument('input_file', help="csv file with data to be split")
    parser.add_argument('output_train', help="output training csv file path")
    parser.add_argument('output_validation', help="output validation csv file path")
    parser.add_argument('output_test', help="output test csv file path")

    args = parser.parse_args()
    main(args.input_file, args.output_train, args.output_validation, args.output_test)

