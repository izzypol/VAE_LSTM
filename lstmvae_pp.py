import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd


# import data 
df1 = pd.read_csv('test_stat974.csv')
df2 = pd.read_csv("test_stat393.csv")
Vstack1_value = df1['Vstack_value'].values
Vstack2_value = df2['Vstack_value'].values
Istack1_value = df1['Istack_value'].values
Istack2_value = df2['Istack_value'].values

# combine the data and convert to tensor
Vstack_value = np.vstack((Vstack1_value, Vstack2_value))
Istack_value = np.vstack((Istack1_value, Istack2_value))
Vstack_value = torch.tensor(Vstack_value, dtype=torch.float32)
Istack_value = torch.tensor(Istack_value, dtype=torch.float32)

# separate the real data in train and test
#train_vstack, test_vstack, train_istack, test_istack = train_test_split(Vstack_value, Istack_value, test_size=0.0)
train_vstack, test_vstack = Vstack_value, Vstack_value

class encoder(nn.Module):
    """
    Le code est séparé en les calcul et le forward 
    """
    def __init__(self, input_size, hidden_size, num_layers):
        """
        fct qui encode l'input en utilisant LSTM
        
        :param input_size: taille de l'input
        :param hidden_size: taille du hidden layer
        :param num_layers: nombre de couches

        return: résultat encodé
        """
        print("encoder", input_size, hidden_size)
        super(encoder, self).__init__()

        # resortir les param
        self.hidden_size = hidden_size
        print("test 1")
        self.num_layers = num_layers
        print("test 2")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        print("test 3")

    def forward(self, x):
        """
        Dfct qui permet de faire le forward pass

        :param x: input (batch_size, seq_len, input_size)
        """
        ouputs, (hidden, cell) = self.lstm(x) # on a pas besoin de outputs mais lstm le sort quand même
        print("encoder forward hidden shape", hidden.shape, "cell", cell.shape) 
        return (hidden, cell)

class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        fct qui decode le latent variable z
        
        :param input_size: taille de l'input -- this isnt necessayr??
        :param hidden_size: taille du hidden layer
        :param output_size: taille de l'output
        :param num_layers: nombre de couches


        return: reconstruction de l'input
        """
        print("decoder", input_size, hidden_size, output_size)
        super(decoder, self).__init__()
        self.hidden_size = hidden_size
        print("test 4")
        self.output_size = output_size
        print("test 5")
        self.num_layers = num_layers
        print("test 6")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        print("test 7 ")
        self.fc = nn.Linear(hidden_size, output_size)
        print("test 8 ")

    def forward(self, x, hidden): 
        """
        fct qui permet de faire le forward pass
        
        :param x: input (batch_size, seq_len, hidden_size)"""

        print("decoder input x", x.shape)

        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)

        print("decoder output shape", output.shape, "prediction shape", prediction.shape)

        return prediction, (hidden, cell)

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, device=None):
        """
        début du model lstmvae
        :param input_size: int batch_size*seq_len*input_dim
        :param hidden_size: int, output size of LSTMAE
        :param latent_size: int, latent z_layer size
        :param device: torch device (cpu or cuda)"""
        super(LSTMVAE, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

        # resortir les params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 1 # peut être varier pour rafiner 
        print("LSTMVAE", input_size, hidden_size, latent_size)
        
        # encodeur
        self.lstm_enc = encoder(
            input_size = input_size, hidden_size = hidden_size, num_layers=self.num_layers)
        print("test 9")
        
        # les deux prochaines lignes sont pour calculer mu et logvar 
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size) # utilise affine linear transf y = x AT + b
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        print("test 10")

        # décodeur
        self.lstm_dec = decoder(
            input_size=latent_size, output_size=input_size, hidden_size=hidden_size, 
            num_layers=self.num_layers)
        print("test 11")
        
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size) 
        self.log_sigma = torch.zeros([])


    def reparameterize(self, mu, logvar):
        """
        fct qui permet de faire le reparameterization trick N(0,1) -> N(mu, var) 
        
        :param mu: moyenne
        :param logvar: log variance

        return: variable aléatoire échantillonnée selon N(mu, var)
        """
        print("reparameterize mu shape", mu.shape, "logvar shape", logvar.shape)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self,x):
        print("LSTMVAE forward input x", x.type)
        batch_size, seq_len, feature_dim = x.shape

        # encoder le input
        enc_hidden, enc_cell = self.lstm_enc(x)
        enc_h = enc_hidden[0].view(batch_size, self.hidden_size).to(self.device)

        # calc mu et logvar
        mu = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparameterize(mu, logvar)

        # init le hidden state avec inputs
        h_ = self.fc3(z)
        c_ = torch.zeros_like(h_)  # Initialize cell state

        # decode le latent space
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)

        # init le hidden state
        hidden = (h_.unsqueeze(0).contiguous(), c_.unsqueeze(0).contiguous())
        recon_output, hidden = self.lstm_dec(z, hidden)

        x_hat = recon_output

        # calc le loss
        loss = self.loss_fct(x_hat, x, mu, logvar)
        m_loss, recon_loss, kld_loss = (
            loss["loss"], 
            loss["recon_loss"],
            loss["KLD"]
        )

        return m_loss, x_hat, (recon_loss, kld_loss)

    def gaussian_nnl(self, mu, log_sigma, x):
        return 0.5*torch.pow((x-mu) / log_sigma.exp(), 2) + log_sigma+ 0.5 * torch.tensor(2*np.pi).log()

    def softclip(self, tensor, min):
        """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
        result_tensor = min + F.softplus(tensor - min)
        return result_tensor
        
    # determine the VAE loss (dif from normal MSE)
    def loss_fct(self, *args) -> dict:
        """
        calc le lorss VAE
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        logvar = args[3]

        # reconstruction loss part
        self.log_sigma = ((input - recons) ** 2).mean().sqrt().log() 
        log_sigma = self.softclip(self.log_sigma, -6)
        recons_loss = self.gaussian_nnl(recons, log_sigma, input).sum()

        # kld loss part
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # total loss
        loss = recons_loss + kld_loss 

        return {'loss': loss, "recon_loss": recons_loss.detach(), "KLD": kld_loss.detach()}

# question: what would be seq len and other stuff for our data!!!!!
# Step 2b: Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # chercher pourquoi cuda fct pas
input_size = 1  # laisse à 1 pcq on a seulement le voltage, mais possiblement 2 dans le futur 
hidden_size = 64 # qu'est-ce qu'on veut mettre pour les données pp? mettre qqch comparable au nombres de points dans la séquence
latent_dim = 5
model = LSTMVAE(input_size, hidden_size, latent_dim, device=device).to(device) # le .to(device) est pour l'efficacité de calcul
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Autoencoder
num_epochs = 500
batch_size = 32
train_vstack_gpu = train_vstack.to(device)  # Pre-move data to device (already 3D)

# save the values of loss, recon loss and kls loss for a graph later
loss_tot = []
recon_tot = []
kld_tot = []

model.train()  # Set to training mode
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    for i in range(0, len(train_vstack_gpu), batch_size):
        batch_noisy = train_vstack_gpu[i:i+batch_size]

        # Forward pass
        loss, x_hat, (recon_loss, kld_loss) = model(batch_noisy)
        loss_tot.append(loss.item())
        recon_tot.append(recon_loss.item())
        kld_tot.append(kld_loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    if (epoch+1) % 10 == 0:
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Step 4: Visualize Results
def visualize_results(model, noisy_data, device, num_samples=100, batch_size=10):
    model.eval()
    with torch.no_grad():

        noisy_test = noisy_data[:3]  # Shape: (3, seq_len, feature_dim)
        batch_size_test, seq_len, feature_dim = noisy_test.shape

        # Encode once to get mu and logvar
        enc_hidden, enc_cell = model.lstm_enc(noisy_test)
        enc_h = enc_hidden[0].view(batch_size_test, model.hidden_size).to(device)
        mu = model.fc21(enc_h)
        logvar = model.fc22(enc_h)
        
        # Sample multiple latent codes efficiently
        samples = []
        for _ in range(num_samples):
            z = model.reparameterize(mu, logvar)
            h_ = model.fc3(z)
            c_ = torch.zeros_like(h_)
            z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
            
            hidden = (h_.unsqueeze(0).contiguous(), c_.unsqueeze(0).contiguous())
            recon_output, _ = model.lstm_dec(z_expanded, hidden)
            samples.append(recon_output)
        
        samples = torch.stack(samples)  # Shape: (num_samples, batch_size, seq_len, feat)
        
        # Compute statistics across samples
        mean_outputs = samples.mean(dim=0)
        std_outputs = samples.std(dim=0)
        lower = mean_outputs - 2 * std_outputs
        upper = mean_outputs + 2 * std_outputs

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[i].plot(vstack_data[i].numpy(), label='Raw data', alpha=0.7)
        axs[i].plot(mean_outputs[i].cpu().numpy(), label='Denoised', linestyle='dashed', linewidth=2)
        axs[i].fill_between(range(seq_len), 
                             lower[i].cpu().numpy(), upper[i].cpu().numpy(),
                             alpha=0.2, color='green', label='±2σ Confidence')
        axs[i].legend()
        axs[i].set_title(f'Sequence {i+1}')

    plt.xlabel('Time Step 0.00001')
    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    axs2.plot(loss_tot, label='Total loss', alpha=0.7)
    axs2.plot(recon_tot, label='Recontruction loss', alpha=0.7)
    axs2.plot(kld_tot, label='KLD loss', alpha=0.7)
    axs2.legend()
    axs2.set_title('Evolution of loss throughout training')

    plt.xlabel('Optimization step')
    plt.tight_layout()
    plt.show()

# Visualize on test and train data (pass both if desired)
visualize_results(model, train_vstack, test_vstack, device=device)