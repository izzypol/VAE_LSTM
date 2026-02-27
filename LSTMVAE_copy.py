import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(50)
np.random.seed(50)

# Step 1: Generate Complex Synthetic Dataset
def generate_complex_sequence(length=50, num_sequences=1000, noise_factor=0.5):
    x = np.linspace(0, 4 * np.pi, length)

    # Create complex clean signal by combining sinusoids of different frequencies
    clean_sequences = np.array([
        np.sin(x + np.random.uniform(0, 2 * np.pi)) +
        0.5 * np.sin(2 * x + np.random.uniform(0, 2 * np.pi)) +
        0.25 * np.sin(4 * x + np.random.uniform(0, 2 * np.pi))
        for _ in range(num_sequences)
    ])

    # Add complex noise: Gaussian noise + occasional spikes + uniform noise
    gaussian_noise = noise_factor * np.random.normal(size=clean_sequences.shape)
    spike_noise = np.random.choice([0, 1], size=clean_sequences.shape, p=[0.98, 0.02]) * np.random.uniform(-3, 3, size=clean_sequences.shape)
    uniform_noise = noise_factor * np.random.uniform(-1, 1, size=clean_sequences.shape)

    noisy_sequences = clean_sequences + gaussian_noise + spike_noise + uniform_noise
    return torch.tensor(noisy_sequences, dtype=torch.float32), torch.tensor(clean_sequences, dtype=torch.float32)

# Generate data
noisy_data, clean_data = generate_complex_sequence()
train_noisy, test_noisy, train_clean, test_clean = train_test_split(noisy_data, clean_data, test_size=0.2)

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

# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # chercher pourquoi cuda fct pas
input_size = 1  # laisse à 1 pcq on a seulement le voltage, mais possiblement 2 dans le futur 
hidden_size = 64 # qu'est-ce qu'on veut mettre pour les données pp? mettre qqch comparable au nombres de points dans la séquence
latent_dim = 5
model = LSTMVAE(input_size, hidden_size, latent_dim, device=device).to(device) # le .to(device) est pour l'efficacité de calcul
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Autoencoder
num_epochs = 500
batch_size = 2
train_vstack_gpu = train_noisy.to(device)  # Pre-move data to device (already 3D)

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
def visualize_results(model, noisy_data, num_samples=100):
    model.eval()
    with torch.no_grad():
        # Sample multiple reconstructions to estimate uncertainty
        samples = []
        for _ in range(num_samples):
            output, _, _ = model(noisy_data)
            samples.append(output)
        samples = torch.stack(samples)  # Shape: (num_samples, batch_size, seq_len)
        
        # Compute mean and standard deviation across samples
        mean_outputs = samples.mean(dim=0)
        std_outputs = samples.std(dim=0)
        
        # Compute lines at mean ± 2*std (approximating 2.5% and 97.5% percentiles for a normal distribution)
        lower = mean_outputs - 2 * std_outputs
        upper = mean_outputs + 2 * std_outputs

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[i].plot(noisy_data[i].numpy(), label='Noisy Input')
        # axs[i].plot(clean_data[i].numpy(), label='Clean Input')
        axs[i].plot(mean_outputs[i].numpy(), label='Mean Denoised Output', linestyle='dashed')
        axs[i].plot(lower[i].numpy(), label='Mean - 2*Std (≈2.5%)', linestyle='dotted', color='red')
        axs[i].plot(upper[i].numpy(), label='Mean + 2*Std (≈97.5%)', linestyle='dotted', color='green')
        axs[i].legend()
        axs[i].set_title(f'Sequence {i+1}')

    plt.xlabel('Time Step 0')
    plt.show()

# Visualize on test data
visualize_results(model, test_noisy)