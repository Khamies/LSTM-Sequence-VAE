import torch

from data.ptb import PTB


class LSTM_VAE(torch.nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers=1, data_dir = "./VGDLData"):
    super(LSTM_VAE, self).__init__()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Variables
    self.num_layers = num_layers
    self.lstm_factor = num_layers
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size

    # For dictionary lookups 
    self.dictionary = PTB(data_dir=data_dir, split="all", create_data= False, max_sequence_length= 600)
  
    # X: bsz * seq_len * vocab_size 
    # Embedding
    self.embed = torch.nn.Embedding(num_embeddings= self.vocab_size,embedding_dim= self.embed_size)

    #    X: bsz * seq_len * vocab_size 
    #    X: bsz * seq_len * embed_size

    # Encoder Part
    self.encoder_lstm = torch.nn.LSTM(input_size= self.embed_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
    self.mean = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)
    self.log_variance = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)

    # Decoder Part
                                        
    self.init_hidden_decoder = torch.nn.Linear(in_features= self.latent_size, out_features= self.hidden_size * self.lstm_factor)
    self.decoder_lstm = torch.nn.LSTM(input_size= self.embed_size, hidden_size= self.hidden_size, batch_first = True, num_layers = self.num_layers)
    self.output = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.vocab_size)
    self.log_softmax = torch.nn.LogSoftmax(dim=2)

  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return (hidden_cell, state_cell)

  def get_embedding(self, x):
    x_embed = self.embed(x)
    
    # Total length for pad_packed_sequence method = maximum sequence length
    maximum_sequence_length = x_embed.size(1)

    return x_embed, maximum_sequence_length

  def encoder(self, packed_x_embed,total_padding_length, hidden_encoder):

    # pad the packed input.

    packed_output_encoder, hidden_encoder = self.encoder_lstm(packed_x_embed, hidden_encoder)
    output_encoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True, total_length= total_padding_length)

    # Extimate the mean and the variance of q(z|x)
    mean = self.mean(hidden_encoder[0])
    log_var = self.log_variance(hidden_encoder[0])
    std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
    
    # Generate a unit gaussian noise.
    batch_size = output_encoder.size(0)
    seq_len = output_encoder.size(1)
    noise = torch.randn(batch_size, self.latent_size).to(self.device)
    
    z = noise * std + mean

    return z, mean, log_var, hidden_encoder


  def decoder(self, z, packed_x_embed, total_padding_length=None):

    hidden_decoder = self.init_hidden_decoder(z)
    hidden_decoder = (hidden_decoder, hidden_decoder)

    # pad the packed input.
    packed_output_decoder, hidden_decoder = self.decoder_lstm(packed_x_embed,hidden_decoder) 
    output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True, total_length= total_padding_length)


    x_hat = self.output(output_decoder)
    
    x_hat = self.log_softmax(x_hat)


    return x_hat

  

  def forward(self, x,sentences_length,hidden_encoder):
    
    """
      x : bsz * seq_len
    
      hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)

    """
    # Get Embeddings
    x_embed, maximum_padding_length = self.get_embedding(x)

    # Packing the input
    packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)


    # Encoder
    z, mean, log_var, hidden_encoder = self.encoder(packed_x_embed, maximum_padding_length, hidden_encoder)

    # Decoder
    x_hat = self.decoder(z, packed_x_embed, maximum_padding_length)
    


    return x_hat, mean, log_var, z, hidden_encoder

  def get_z(self, words, sentences_length):
    indexes = [self.dictionary.get_w2i().get(word, self.dictionary.w2i['<unk>']) for word in words]
    x = torch.IntTensor([indexes])
    x_hat, mean, log_var, z, hidden_encoder = self.forward(x, torch.IntTensor([len(words)]),self.init_hidden(1) )

    return z
  

  def inference(self, n_samples, sos, z, end_early = True):

    # generate random z 
    batch_size = 1
    seq_len = 1
    idx_sample = []


    input = torch.Tensor(1, 1).fill_(self.dictionary.get_w2i()[sos]).long().to(self.device)

    hidden = self.init_hidden_decoder(z)
    hidden = (hidden, hidden)
    
    for i in range(n_samples):
      input = self.embed(input)
      output,hidden = self.decoder_lstm(input, hidden)
      output = self.output(output)
      output = self.log_softmax(output)
      output = output.exp()
      _, s = torch.topk(output, 1)
      if end_early and  (self.dictionary.get_i2w()[str(s.item())]  == "<eos>"):
        break 
      else:
        idx_sample.append(s.item())
        input = s.squeeze(0)

    w_sample = [self.dictionary.get_i2w()[str(idx)] for idx in idx_sample]
    w_sample = " ".join(w_sample)

    return w_sample
