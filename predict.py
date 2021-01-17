from tensorflow.keras.models import load_model
import numpy as np
import sys
import pandas as pd
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file path', required=True)
    parser.add_argument('-o', '--output', help='Output file path', default='output.csv')
    args = parser.parse_args()
    return (args.input, args.output)

def load_file(path):
    f = pd.read_csv(path, header=None).to_numpy()
    f = f.reshape(-1, 32, 32, 1)
    f = f / 255.0
    return f

def main():
    inp, outp = parse_args()
    try:
        model = load_model('model.keras')
    except Exception as e:
        print('Unable to load model. Make sure file model.keras is in root directory: ', e)
        sys.exit()
    try:
        f = load_file(inp)
    except Exception as e:
        print('Error druring opening input file: ', e)
        sys.exit()
    try:
        prediction = model.predict(f)
    except Exception as e:
        print('Error during prediction: ', e)
        sys.exit()
    pred_classes = np.argmax(prediction, axis=1)
    try:
        np.savetxt(outp, pred_classes, delimiter=',', fmt='%d')
    except Exception as e:
        print('Error during saving results: ', e)
        sys.exit()

main()