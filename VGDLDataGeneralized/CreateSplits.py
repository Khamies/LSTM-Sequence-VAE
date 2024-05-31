
import csv
import random 
import math 


if __name__ == "__main__":
    all_files = []
    percent_train = 0.8 
    percent_valid = 0.1 
    percent_test = 0.1 
      
    with open("VGDLDataGeneralized/examples/all_games_sp.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            all_files.append(row[1])
    random.shuffle(all_files)


    train_index = round(len(all_files) * percent_train)
    test_index = train_index + round(len(all_files) * percent_test)


    training_files = all_files[:train_index]
    test_files = all_files[train_index:test_index]
    valid_files = all_files[test_index:]

    with open("VGDLDataGeneralized/ptb.all.csv", "w") as file:
        cvs_writer = csv.writer(file, delimiter=",")
        for i, f in enumerate(all_files):
            cvs_writer.writerow([i, f])

    with open("VGDLDataGeneralized/ptb.train.csv", "w") as file:
        cvs_writer = csv.writer(file, delimiter=",")
        for i, f in enumerate(training_files):
            cvs_writer.writerow([i, f])

    with open("VGDLDataGeneralized/ptb.test.csv", "w") as file:
        cvs_writer = csv.writer(file, delimiter=",")
        for i, f in enumerate(test_files):
            cvs_writer.writerow([i, f])

    with open("VGDLDataGeneralized/ptb.valid.csv", "w") as file:
        cvs_writer = csv.writer(file, delimiter=",")
        for i, f in enumerate(valid_files):
            cvs_writer.writerow([i, f])



