import torch
from VGDLData.ptb import PTB
from model import LSTM_VAE 
batch_size = 20
bptt = 200
lr = 0.001

embed_size = 50
hidden_size = 200
latent_size = 40
lstm_layer=1
device = "cpu"

# Load the data
train_data = PTB(data_dir="VGDLData", split="train", create_data= False, max_sequence_length= bptt)
test_data = PTB(data_dir="VGDLData", split="test", create_data= False, max_sequence_length=bptt)
valid_data = PTB(data_dir="VGDLData", split="valid", create_data= False, max_sequence_length= bptt)


# Batchify the data
train_loader = torch.utils.data.DataLoader( dataset= train_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader( dataset= test_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())
valid_loader = torch.utils.data.DataLoader( dataset= valid_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())

vocab_size = train_data.vocab_size
model = LSTM_VAE(vocab_size = vocab_size, embed_size = embed_size, hidden_size = hidden_size, latent_size = latent_size).to(device)

checkpoint = torch.load("models/VGDL_VAE.pt", map_location=torch.device('cpu') )
model.load_state_dict(checkpoint)

pytorch_total_params = sum(p.numel() for p in model.parameters())

print(pytorch_total_params)