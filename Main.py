from VGDLDataGeneralized.ptb import PTB

import torch
from loss import VAE_Loss
from model import LSTM_VAE
from train import Trainer

from settings import global_setting, model_setting, training_setting

from utils import  interpolate, plot_elbo, get_latent_codes, visualize_latent_codes

import argparse

# General Settings

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(global_setting["seed"])


"""parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
parser.add_argument("--batch_size", type=str, default="20")
parser.add_argument("--bptt", type=str,default="600")
parser.add_argument("--embed_size", type=str, default="50") 
parser.add_argument("--hidden_size", type=str, default="400")
parser.add_argument("--latent_size", type=str, default="60")
parser.add_argument("--lr", type=str, default="0.001")


# Extract commandline arguments   
args = parser.parse_args()"""

batch_size = training_setting["batch_size"]
bptt =   training_setting["bptt"]
embed_size =  model_setting["embed_size"]
hidden_size = model_setting["hidden_size"]
latent_size = model_setting["latent_size"]
lr = training_setting["lr"]

data_dir = "./VGDLDataGeneralized"

# Load the data
train_data = PTB(data_dir=data_dir, split="train", create_data= False, max_sequence_length= bptt)
test_data = PTB(data_dir=data_dir, split="test", create_data= False, max_sequence_length=bptt)
valid_data = PTB(data_dir=data_dir, split="valid", create_data= False, max_sequence_length= bptt)

# Batchify the data
train_loader = torch.utils.data.DataLoader( dataset= train_data, batch_size=batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader( dataset= test_data, batch_size= batch_size, shuffle= True)
valid_loader = torch.utils.data.DataLoader( dataset= valid_data, batch_size= batch_size, shuffle= True)



vocab_size = train_data.vocab_size
model = LSTM_VAE(vocab_size = vocab_size, embed_size = embed_size, hidden_size = hidden_size, latent_size = latent_size, data_dir=data_dir).to(device)

Loss = VAE_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr= training_setting["lr"])

trainer = Trainer(train_loader, test_loader, model, Loss, optimizer)


def main():
    # Epochs
    train_losses = []
    test_losses = []
    for epoch in range(training_setting["epochs"]):
        print("Epoch: ", epoch)
        print("Training.......")
        train_losses = trainer.train(train_losses, epoch, training_setting["batch_size"], training_setting["clip"])
        print("Testing.......")
        test_losses = trainer.test(test_losses, epoch, training_setting["batch_size"])
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "models/VGDL_VAE_GENERALIZED2_" + str(epoch) + ".pt")


    plot_elbo(train_losses, "train")
    plot_elbo(test_losses, "test")

    torch.save(model.state_dict(), "models/VGDL_VAE2.pt")

if __name__ == "__main__":

    main()



