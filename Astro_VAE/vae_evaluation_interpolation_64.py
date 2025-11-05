import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

# --- HYPERPARAMETERS (Must match your trained model) ---
LATENT_DIM = 128
IMAGE_CHANNELS = 4
TARGET_SIZE = 64
ORIGINAL_SIZE = 108
BASE_CHANNELS = 32
IMAGE_MEAN = [0.5] * IMAGE_CHANNELS
IMAGE_STD = [0.5] * IMAGE_CHANNELS
MODEL_SAVE_PATH = 'vae_checkpoint_4ch_64patch.pth' # Path to your trained weights
DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5'

# ======================================================================
# CORE MODEL AND DATA UTILITIES (Simplified definitions from training script)
# ======================================================================

# --- Placeholder/Required Data Structure ---
class NumPyVAEDataset(Dataset):
    """Placeholder for the actual dataset class used during training."""
    def __init__(self, data_array: np.ndarray):
        self.data = data_array.astype(np.float32)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
        self.start_index = (ORIGINAL_SIZE - TARGET_SIZE) // 2
        self.end_index = self.start_index + TARGET_SIZE

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_tensor = torch.from_numpy(self.data[idx])
        # Central Crop (64x64 from 108x108)
        image_tensor_cropped = image_tensor[:, self.start_index:self.end_index, self.start_index:self.end_index]
        image_tensor_cropped = self.transform(image_tensor_cropped)
        return image_tensor_cropped

def percentile_normalize_channel_wise(data_array_raw, low_percentile=10, high_percentile=99.7):
    """Performs channel-wise percentile scaling across the entire dataset."""
    # Logic is assumed correct from previous step
    data_array = data_array_raw[:, 1:5, :, :].astype(np.float32)
    N, C, H, W = data_array.shape
    reshaped_data = data_array.transpose(1, 0, 2, 3).reshape(C, -1)
    
    p_low = np.percentile(reshaped_data, low_percentile, axis=1)
    p_high = np.percentile(reshaped_data, high_percentile, axis=1)

    p_low_broadcast = p_low.reshape(1, C, 1, 1)
    p_high_broadcast = p_high.reshape(1, C, 1, 1)
    
    data_range = p_high_broadcast - p_low_broadcast
    data_range[data_range == 0] = 1e-10 
    
    data_array_norm = np.clip(data_array, p_low_broadcast, p_high_broadcast)
    data_array_norm = (data_array_norm - p_low_broadcast) / data_range
    data_array_norm[np.isnan(data_array_norm)] = 0.0
    
    return data_array_norm

class ConvolutionalVAE(nn.Module):
    """The VAE architecture class (structure must match the trained model)."""
    def __init__(self, target_size=TARGET_SIZE, latent_dim=LATENT_DIM, 
                 in_channels=IMAGE_CHANNELS, base_channels=BASE_CHANNELS):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = 3 # 64x64 -> 8x8 requires 3 layers
        self.FINAL_SPATIAL_SIZE = 8
        
        encoder_channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        
        # --- Encoder Layers ---
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_layers.append(
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=4, stride=2, padding=1)
            )
        
        FINAL_CHANNELS = encoder_channels[-1] # 128
        self.flatten_size = FINAL_CHANNELS * self.FINAL_SPATIAL_SIZE * self.FINAL_SPATIAL_SIZE

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # --- Decoder Layers ---
        self.fc_z = nn.Linear(latent_dim, self.flatten_size)
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = encoder_channels[self.num_layers - i]
            out_ch = encoder_channels[self.num_layers - i - 1]
            self.decoder_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        x = x.view(x.size(0), -1) 
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        x = self.fc_z(z)
        final_encoder_channels = self.encoder_layers[-1].out_channels
        x = x.view(x.size(0), final_encoder_channels, self.FINAL_SPATIAL_SIZE, self.FINAL_SPATIAL_SIZE) 
        
        for i, layer in enumerate(self.decoder_layers):
            if i == self.num_layers - 1:
                reconstructed_x = torch.tanh(layer(x))
            else:
                x = F.relu(layer(x))
        
        return reconstructed_x, mu, logvar, z

# ======================================================================
# 4. INTERPOLATION AND VISUALIZATION LOGIC
# ======================================================================

def interpolate_latent_space(model, z1, z2, num_steps, device):
    """
    Interpolates linearly between two latent vectors and reconstructs the images.
    """
    # Create an array of interpolation factors (0.0 to 1.0)
    alphas = np.linspace(0.0, 1.0, num_steps)
    
    # Pre-allocate array for reconstructed images
    reconstructions = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation: z_interp = (1 - alpha) * z1 + alpha * z2
            z_interp = (1.0 - alpha) * z1 + alpha * z2
            
            # Pass the interpolated latent vector through the decoder
            x = model.fc_z(z_interp)
            
            # Reshape back to feature map
            final_encoder_channels = model.encoder_layers[-1].out_channels
            x = x.view(x.size(0), final_encoder_channels, model.FINAL_SPATIAL_SIZE, model.FINAL_SPATIAL_SIZE) 

            # Run through ConvTranspose layers
            for i, layer in enumerate(model.decoder_layers):
                if i == model.num_layers - 1:
                    reconstructed_x = torch.tanh(layer(x))
                else:
                    x = F.relu(layer(x))
            
            reconstructions.append(reconstructed_x.cpu().squeeze(0))

    return reconstructions

def plot_interpolation_sequence(reconstructions, filename='interpolation_sequence.png', images_per_row=10):
    """ Plots the sequence of reconstructed images in a grid. """
    
    # Setup 3-channel denormalization for visualization
    mean_3ch = IMAGE_MEAN[:3]
    std_3ch = IMAGE_STD[:3]
    inv_normalize_3ch = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
        std=[1/s for s in std_3ch]
    )

    num_steps = len(reconstructions)
    num_rows = int(np.ceil(num_steps / images_per_row))
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 1.5, num_rows * 1.5))
    axes = axes.flatten()

    for i in range(num_steps):
        img_tensor = reconstructions[i]
        
        # Denormalize and convert to HWC (use first 3 channels)
        img_np = inv_normalize_3ch(img_tensor[:3, :, :]).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        ax = axes[i]
        ax.imshow(img_np)
        ax.set_title(f"{i}", fontsize=8)
        ax.axis('off')

    # Turn off any unused axes
    for j in range(num_steps, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Latent Interpolation Sequence ({num_steps} Steps)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Interpolation sequence plot saved to {filename}")


# ======================================================================
# 5. MAIN EXECUTION
# ======================================================================

if __name__ == '__main__':
    
    # --- Setup ---
    DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5' 
    INTERPOLATION_STEPS = 10
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- 1. Data Loading ---
    print(f"Loading data from {DATA_FILE_PATH}...")
    try:
        with h5py.File(DATA_FILE_PATH, 'r') as hf:
            data_array_raw = hf['images'][:] 
            # Only need two random indices for interpolation
            random_indices = np.random.choice(data_array_raw.shape[0], 2, replace=False)
    except Exception as e:
        print(f"FATAL ERROR: Could not load HDF5 data: {e}. Exiting.")
        raise SystemExit(1)

    # Preprocess the data array once
    preprocessed_data_array = percentile_normalize_channel_wise(data_array_raw, low_percentile=10, high_percentile=99.7)
    
    # Create a DataLoader just to get the data efficiently (we won't use the whole loop)
    data_loader = DataLoader(NumPyVAEDataset(preprocessed_data_array), batch_size=2, shuffle=False)
    
    # --- 2. Model Loading ---
    print(f"\nLoading trained model from {MODEL_SAVE_PATH}...")
    model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(torch.float32).to(device)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Checkpoint file not found at {MODEL_SAVE_PATH}. Cannot perform evaluation.")
        raise SystemExit(1)
        
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # --- 3. Extract Latent Codes for Two Images ---
    
    # Get the two specific images (Index 0 and 1 from the data_loader batch)
    try:
        images_batch = next(iter(data_loader)).to(torch.float32).to(device)
        image1 = images_batch[0].unsqueeze(0) # Keep batch dim
        image2 = images_batch[1].unsqueeze(0)

        # Extract mu (mean) for the start (z1) and end (z2) points
        _, mu1, _, _ = model(image1)
        _, mu2, _, _ = model(image2)
        
        # Use the mu vectors as the reliable latent codes for interpolation
        z1 = mu1
        z2 = mu2
        
    except Exception as e:
        print(f"Error during feature extraction: {e}. Cannot proceed.")
        raise SystemExit(1)


    # --- 4. Perform Interpolation and Plot ---
    print(f"Performing {INTERPOLATION_STEPS} steps of latent space interpolation...")
    reconstructions = interpolate_latent_space(model, z1, z2, INTERPOLATION_STEPS, device)
    
    plot_interpolation_sequence(reconstructions, images_per_row=10)
    
    print("\nLatent space interpolation successfully visualized.")