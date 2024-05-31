from ptb import PTB 
if __name__ == "__main__":
    data_dir = "VGDLDataGeneralized"

    all = PTB(data_dir, "all",create_data=True)
    training = PTB(data_dir, "train",create_data=True)
    test = PTB(data_dir, "test",create_data=True)
    valid = PTB(data_dir, "valid",create_data=True)