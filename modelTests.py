import torch
from VGDLData.ptb import PTB
from model import LSTM_VAE


batch_size = 20
bptt = 200
lr = 0.001

embed_size = 50
hidden_size = 1000
latent_size = 60
lstm_layer=1
device = "cpu"

data_dir = "VGDLDataGeneralized"

# Load the data
train_data = PTB(data_dir=data_dir, split="train", create_data= False, max_sequence_length= bptt)
test_data = PTB(data_dir=data_dir, split="test", create_data= False, max_sequence_length=bptt)
valid_data = PTB(data_dir=data_dir, split="valid", create_data= False, max_sequence_length= bptt)


# Batchify the data
train_loader = torch.utils.data.DataLoader( dataset= train_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader( dataset= test_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())
valid_loader = torch.utils.data.DataLoader( dataset= valid_data, batch_size= batch_size, shuffle= True, pin_memory=torch.cuda.is_available())

def interpolate(model, n_interpolations, sos, sequence_length):

  # # Get input.

  z1 = torch.randn((1,1,latent_size)).to(device)
  z2 = torch.randn((1,1,latent_size)).to(device)

  text1 = model.inference(sequence_length , sos, z1)
  text2 = model.inference(sequence_length , sos, z2)

  alpha_s = torch.linspace(0,1,n_interpolations)

  interpolations = torch.stack([alpha*z1 + (1-alpha)*z2  for alpha in alpha_s])


  samples = [model.inference(sequence_length , sos, z) for z in interpolations]




  return samples, text1, text2

vocab_size = train_data.vocab_size
model = LSTM_VAE(vocab_size = vocab_size, embed_size = embed_size, hidden_size = hidden_size, latent_size = latent_size, data_dir=data_dir).to(device)

checkpoint = torch.load("models/VGDL_VAE_GENERALIZED.pt", map_location=torch.device('cpu') )
model.load_state_dict(checkpoint)

#@title Sample Generation
# inference
z1 = torch.randn(1,1,latent_size).to(device)
print(z1)
z2 = torch.randn(1,1,latent_size).to(device)
print(z2)

sos = "<sos>"
sample1 = model.inference(600 , sos, z1)
sample2 = model.inference(600 , sos , z2)


words = train_data._file_to_words("VGDLDataGeneralized/examples/gridphysics/aliens.txt") 

z = model.get_z(words, 600) 
aliens = model.inference(600 , sos, z)
print(aliens)


samples, text1, text2 = interpolate(model,3,sos, 600)
print("First sentence:", text1)
print("Second sentence:", text2)

for sample in samples: print(sample)