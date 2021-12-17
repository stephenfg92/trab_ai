import struct
from PIL import Image
import pandas as pd
import numpy as np

DATA_DIR = 'data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'

LBL_A = 'atk'
LBL_B = 'norm'

path = '../resumo.csv'
df = pd.read_csv(path,
                 sep = ",",
                 encoding = "UTF-8")

df_a = df[df.attack == 1]
df_b = df[df.attack == 0]

def split_into_img(df, ratio):
    df_len = len(df)
    train_len = ratio * df_len
    val_test_len = 1 - train_len
    for i in range(0, df_len):
        data = df.iloc[i]
        line = (np.array(data[:-1]).tobytes(), int(data[-1]))
        size = struct.pack("<I", len(line[0]))
        image_side = int( ( ( len(line[0]) + len(size) ) / 3.0 ) ** 0.5) + 1
        img = Image.frombytes("RGB", 
                             (image_side, image_side),
                             size + line[0] + b"\x00" * (image_side ** 2 * 3 - ( len(size) + len(line[0]) ))
                             )

        FP_STRING = None
        if i < train_len:
            FP_STRING = TRAIN_DIR
        else:
            FP_STRING = TEST_DIR

        if line[1] == 0:
            FP_STRING += LBL_B + '/' + str(i) + '_' + LBL_B + ".png"
        else:
            FP_STRING += LBL_A + '/' + str(i) + str(i) + '_' + LBL_A + ".png"

        img.save(FP_STRING)
        print("\n " + FP_STRING + ' salvo!')

ratio = (70 / 100)
split_into_img(df_a, ratio)
split_into_img(df_b, ratio)

f = open(DATA_DIR + "balance.txt", "w")
f.write("{}".format(len(df_a)))
f.write("\n")
f.write("{}".format(len(df_b)))
f.close()
print("Criado aquivo balance.txt contendo informações sobre a distribuição das classes.")

print("\nFIM!")