# Copy some files from the source directory to the destination directory
# Usage: python3 helper.py <source directory> <destination directory> -n <number of files>

import os
import sys
import shutil
import random

def copy_files(sourceDir, destDir, numFiles):
    files = os.listdir(sourceDir)
    if not os.path.exists(destDir):
        os.mkdir(destDir)
    random.shuffle(files)
    for i in range(numFiles):
        shutil.copy2(os.path.join(sourceDir, files[i]), destDir)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 helper.py <source directory> <destination directory> -n <number of files>")
        sys.exit(1)
    sourceDir = sys.argv[1]
    destDir = sys.argv[2]
    numFiles = int(sys.argv[4])
    copy_files(sourceDir, destDir, numFiles)
