# Rename license plate pictures whose names are Chinese characters to numbers with format 'x_x_x_x_x_x_x.jpg'
# Usage: python lpcha2num.py -d <path to license plate pictures> -r <path to resume file>

import os
import argparse

ctc_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ctc_ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ctc_table = ctc_provinces + ctc_ads

def lpcha2num(path, resume):
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    if not os.path.exists(resume):
        os.mkdir(resume)
    files = os.listdir(path)
    for file in files:
        if file.endswith(".jpg"):
            name = file.split(".")[0]
            if len(name) != 7:
                continue
            newname = ""
            for i in range(7):
                newname += str(ctc_table.index(name[i]))
                if i != 6:
                    newname += "_"
            # Copy file to resume directory
            os.system("copy " + os.path.join(path, file) + " " + os.path.join(resume, newname + ".jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Path to license plate pictures", type=str, required=True)
    parser.add_argument("-r", "--resume", help="Path to resume file", type=str, required=True)
    args = parser.parse_args()
    lpcha2num(args.dir, args.resume)

