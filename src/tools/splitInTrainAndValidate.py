import sys
import os
import random

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print("usage: splitInTrainAndValidate.py ANNOTATIONS_FILE|IMAGE_FOLDER OUT_DIR PERCENT_TRAIN(FLOAT) -darknet (OPTIONAL)")
    
    else:
        darknet = len(sys.argv) == 5 and sys.argv[4] == "-darknet"

        if not os.path.exists(sys.argv[2]): os.mkdir(sys.argv[2])

        valPath = os.path.join(sys.argv[2], "val.txt")
        trainPath = os.path.join(sys.argv[2], "train.txt")

        valAmount = 0
        trainAmount = 0

        with open(valPath, "w+") as val, open(trainPath, "w+") as train:
            if darknet:
                for file in os.listdir(sys.argv[1]):
                    rand = random.random()
                    if rand >= 0 and rand <= float(sys.argv[3]) / 100:
                        train.write(os.path.basename(file) + "\n")
                        trainAmount += 1
                    else:
                        val.write(os.path.basename(file) + "\n")
                        valAmount += 1
            else:
                with open(sys.argv[1], "r") as file:
                    for line in file:
                        rand = random.random()
                        if rand >= 0 and rand <= float(sys.argv[3]) / 100:
                            train.write(line)
                            trainAmount += 1
                        else:
                            val.write(line)
                            valAmount += 1

        print(f"Saved splits (Training: {float(sys.argv[3])}%):\nTrain: {trainPath} Amount: {trainAmount}\nValidate:{valPath} Amount: {valAmount}")
