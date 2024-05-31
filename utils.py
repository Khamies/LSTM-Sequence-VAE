import matplotlib.pyplot as plt
from data.ptb import PTB
from settings import training_setting
import torch


def get_batch(batch):
  sentences = batch["input"]
  target = batch["target"]
  sentences_length = batch["length"]

  return sentences, target, sentences_length


def plot_elbo(losses, mode):
    elbo_loss = list(map(lambda x: x[0], losses))
    kl_loss = list(map(lambda x: x[1], losses))
    recon_loss = list(map(lambda x: x[2], losses))

    losses = {"elbo": elbo_loss, "kl": kl_loss, "recon": recon_loss}
    print(losses)
    """for key in losses.keys():
        plt.plot(losses.get(key), label=key+"_" + mode)

    plt.legend()
    plt.show()"""


def get_latent_codes(dataloader, model, batch_size):
  hidden = model.init_hidden(batch_size)

  Z = []

  with torch.no_grad():
    for batch in dataloader:

        x, t, leng = batch.get("input"), batch.get(
            "target"), batch.get("length")
        x = x.to(model.device)
        t.to(model.device)
        _, _, _, z, _ = model(x, leng, hidden)
        Z.append(z)

    Z = torch.cat(Z[:-1])
    Z = Z.reshape(-1, Z.size(2))
    return Z


def visualize_latent_codes(z):


    z = z.squeeze(0).t().contiguous()

    n_z = z.size(0)
    n = n_z//2

    fig = plt.figure(figsize=(20, 6), dpi=80)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 2*n):

        ax = fig.add_subplot(2, n, i)
        ax.hist(z[i].tolist())

    plt.show()


def interpolate(model, n_interpolations, sos, sequence_length):


  z1 = torch.randn((1,1,model.latent_size)).to(model.device)
  z2 = torch.randn((1,1,model.latent_size)).to(model.device)

  text1 = model.inference(sequence_length , sos, z1)
  text2 = model.inference(sequence_length , sos, z2)

  alpha_s = torch.linspace(0,1,n_interpolations)

  interpolations = torch.stack([alpha*z1 + (1-alpha)*z2  for alpha in alpha_s])


  samples = [model.inference(sequence_length , sos, z) for z in interpolations]




  return samples, text1, text2

