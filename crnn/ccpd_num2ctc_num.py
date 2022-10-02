# Rename license plate pictures whose names is format 'x_x_x_x_x_x_x.jpg' from CCPD to CTC format
# Usage: python ccpd_num2ctc_num.py -d <path to CCPD license plate pictures> -r <path to resume file>

import os
import argparse
import shutil

# CCPD license plate Tabel
provNum, alphaNum, adNum = 38, 25, 35
ccpd_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ccpd_alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ccpd_ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

ccpd_provinces_dict = {0: "皖",
                       1: "沪",
                       2: "津",
                       3: "渝",
                       4: "冀",
                       5: "晋",
                       6: "蒙",
                       7: "辽",
                       8: "吉",
                       9: "黑",
                       10: "苏",
                       11: "浙",
                       12: "京",
                       13: "闽",
                       14: "赣",
                       15: "鲁",
                       16: "豫",
                       17: "鄂",
                       18: "湘",
                       19: "粤",
                       20: "桂",
                       21: "琼",
                       22: "川",
                       23: "贵",
                       24: "云",
                       25: "藏",
                       26: "陕",
                       27: "甘",
                       28: "青",
                       29: "宁",
                       30: "新",
                       31: "警",
                       32: "学",
                       33: "O"}

ccpd_alphabets_dict = {0: "A",
                       1: "B",
                       2: "C",
                       3: "D",
                       4: "E",
                       5: "F",
                       6: "G",
                       7: "H",
                       8: "J",
                       9: "K",
                       10: "L",
                       11: "M",
                       12: "N",
                       13: "P",
                       14: "Q",
                       15: "R",
                       16: "S",
                       17: "T",
                       18: "U",
                       19: "V",
                       20: "W",
                       21: "X",
                       22: "Y",
                       23: "Z",
                       24: "O"}

ccpd_ads_dict = {0: "A",
                 1: "B",
                 2: "C",
                 3: "D",
                 4: "E",
                 5: "F",
                 6: "G",
                 7: "H",
                 8: "J",
                 9: "K",
                 10: "L",
                 11: "M",
                 12: "N",
                 13: "P",
                 14: "Q",
                 15: "R",
                 16: "S",
                 17: "T",
                 18: "U",
                 19: "V",
                 20: "W",
                 21: "X",
                 22: "Y",
                 23: "Z",
                 24: "0",
                 25: "1",
                 26: "2",
                 27: "3",
                 28: "4",
                 29: "5",
                 30: "6",
                 31: "7",
                 32: "8",
                 33: "9",
                 34: "O"}

# CTC license plate Tabel
ctc_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ctc_ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ctc_table = ctc_provinces + ctc_ads

ctc_table_dict = {0: "皖",
                  1: "沪",
                  2: "津",
                  3: "渝",
                  4: "冀",
                  5: "晋",
                  6: "蒙",
                  7: "辽",
                  8: "吉",
                  9: "黑",
                  10: "苏",
                  11: "浙",
                  12: "京",
                  13: "闽",
                  14: "赣",
                  15: "鲁",
                  16: "豫",
                  17: "鄂",
                  18: "湘",
                  19: "粤",
                  20: "桂",
                  21: "琼",
                  22: "川",
                  23: "贵",
                  24: "云",
                  25: "藏",
                  26: "陕",
                  27: "甘",
                  28: "青",
                  29: "宁",
                  30: "新",
                  31: "警",
                  32: "学",
                  33: "0",
                  34: "1",
                  35: "2",
                  36: "3",
                  37: "4",
                  38: "5",
                  39: "6",
                  40: "7",
                  41: "8",
                  42: "9",
                  43: "A",
                  44: "B",
                  45: "C",
                  46: "D",
                  47: "E",
                  48: "F",
                  49: "G",
                  50: "H",
                  51: "I",
                  52: "J",
                  53: "K",
                  54: "L",
                  55: "M",
                  56: "N",
                  57: "P",
                  58: "Q",
                  59: "R",
                  60: "S",
                  61: "T",
                  62: "U",
                  63: "V",
                  64: "W",
                  65: "X",
                  66: "Y",
                  67: "Z",
                  68: "O"}


def ccpd_num2ctc_num(path, resume):
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    if not os.path.exists(resume):
        os.mkdir(resume)
    files = os.listdir(path)
    for file in files:
        if file.endswith(".jpg"):
            name = file.split(".")[0]
            name = name.split("_")
            if len(name) != 7:
                print("File name format error!")
                continue
            new_name = ""
            new_name += str(ctc_table.index(ccpd_provinces_dict[int(name[0])])) + "_"
            for i in range(1, 7):
                new_name += str(ctc_table.index(ccpd_ads_dict[int(name[i])]))
                if i != 6:
                    new_name += "_"
            new_name += ".jpg"
            shutil.copy(os.path.join(path, file), os.path.join(resume, new_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='path to CCPD dataset', type=str, required=True)
    parser.add_argument('-r', '--resume', help='path to resume', type=str, required=True)
    args = parser.parse_args()
    ccpd_num2ctc_num(args.dir, args.resume)
