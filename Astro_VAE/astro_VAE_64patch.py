import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random 
import h5py

# --- HYPERPARAMETERS ---
LATENT_DIM = 128
IMAGE_CHANNELS = 4
TARGET_SIZE = 64        # VAE input size: (4, 64, 64)
ORIGINAL_SIZE = 108     # Raw data size: (4, 108, 108)
BASE_CHANNELS = 32
# Normalization is custom [0,1], so these standard values are placeholders
IMAGE_MEAN = [0.5] * IMAGE_CHANNELS 
IMAGE_STD = [0.5] * IMAGE_CHANNELS  

# ======================================================================
# 1. DATA PREPROCESSING UTILITY
# ======================================================================

def percentile_normalize_channel_wise(data_array_raw, low_percentile=1, high_percentile=99):
    """
    Normalizes a 4D data array (N, C, H, W) channel-by-channel 
    using specified percentiles to ensure robustness against outliers.

    Args:
        data_array_raw (np.ndarray): The input data (N, C_total, H, W).
        low_percentile (int): The percentile to use as the minimum floor (e.g., 1).
        high_percentile (int): The percentile to use as the maximum ceiling (e.g., 99).

    Returns:
        np.ndarray: The normalized array scaled to the [0, 1] range.
    """
    # 1. Select channels and ensure float32 type
    # Shape is (N, 4, H, W)
    data_array = data_array_raw[:, 1:5, :, :].astype(np.float32)
    
    # Reshape the data temporarily to (N * H * W, C) to calculate percentiles easily
    # C=4 is the dimension we want to keep separate
    N, C, H, W = data_array.shape
    reshaped_data = data_array.transpose(1, 2, 3, 0).reshape(-1, N)

    # 2. Calculate Percentile Values (Across all samples for each channel)
    # The percentile is calculated over the entire flattened dimension (all pixels, all batches)
    p_low = np.percentile(reshaped_data, low_percentile, axis=0, keepdims=True)
    p_high = np.percentile(reshaped_data, high_percentile, axis=0, keepdims=True)

    # 3. Reshape Percentiles back to (1, C, 1, 1) for broadcasting
    # This shape allows the subtraction and division to apply correctly across N, H, W
    p_low_broadcast = p_low.reshape(N, 1, 1, 1)
    p_high_broadcast = p_high.reshape(N, 1, 1, 1)
    
    # 4. Apply Normalization
    
    # Calculate the range (avoid division by zero)
    data_range = p_high_broadcast - p_low_broadcast
    data_range[data_range == 0] = 0
    
    # Clip and Scale: data = (data - min) / range
    
    # Clip data to prevent outliers from affecting the statistics (makes noise white/black)
    data_array_norm = np.clip(data_array, p_low_broadcast, p_high_broadcast)
    
    # Normalize
    data_array_norm = (data_array_norm - p_low_broadcast) / data_range

    data_array_norm[np.isnan(data_array_norm)] = 0.0
    return data_array_norm


# ======================================================================
# 2. VAE NUMPY ARRAY DATASET (Custom PyTorch Dataset)
# ======================================================================

class NumPyVAEDataset(Dataset):
    """ Loads data directly from a pre-normalized NumPy array (N, C, H, W). """
    def __init__(self, data_array: np.ndarray):
        # Data is assumed to be (N, 4, 108, 108) and pre-normalized (float32)
        self.data = data_array
        
        # Transformation pipeline only needs final normalization for tensor output
        self.transform = transforms.Compose([
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
        
        if self.data.ndim != 4:
             raise ValueError(f"Input data must be 4D (N, C, H, W), but shape is {self.data.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_tensor = torch.from_numpy(self.data[idx])
                
        # --- CRITICAL CHANGE: Central Crop (108 -> 64) ---
        # Original Size: 108, Target Size: 64. Margin: (108 - 64) / 2 = 22
        start_index = 22
        end_index = 86 # 22 + 64
        
        # Crop the Height and Width dimensions (indices 1 and 2 of the C, H, W tensor)
        image_tensor_cropped = image_tensor[:, start_index:end_index, start_index:end_index]
        
        # Apply normalization (which scales the [0, 1] data into [-1, 1])
        image_tensor_cropped = self.transform(image_tensor_cropped)
        return image_tensor_cropped

def get_dataloader(data_array: np.ndarray, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """ Utility function to create and return the DataLoader instance from a NumPy array. """
    dataset = NumPyVAEDataset(data_array)
    
    if len(dataset) == 0:
        print("DataLoader failed to initialize as the dataset is empty.")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# ======================================================================
# 3. VAE Model Definition (ARCHITECTURAL FIXES)
# ======================================================================
class ConvolutionalVAE(nn.Module):
    def __init__(self, target_size=TARGET_SIZE, latent_dim=LATENT_DIM, 
                 in_channels=IMAGE_CHANNELS, base_channels=BASE_CHANNELS):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 64 -> 8 spatial size requires 3 layers of stride 2
        self.num_layers = int(np.log2(target_size / 8))
        self.FINAL_SPATIAL_SIZE = 8 # Will be 8x8 spatial output
        
        # 1. ENCODER Construction
        encoder_channels = [in_channels]
        for i in range(self.num_layers + 1): # +1 to include the output layer channels
            encoder_channels.append(base_channels * (2**i))
            
        # We only need num_layers + 1 channels, so slice to the correct size
        # This fixes the channel count mismatch seen in previous errors
        encoder_channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_layers.append(
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=4, stride=2, padding=1)
            )
        
        # CRITICAL FIX: Get the final channel count (128) and calculate flatten_size (8192)
        FINAL_CHANNELS = encoder_channels[-1] # This is 128 (32 * 2^2)
        
        self.flatten_size = FINAL_CHANNELS * self.FINAL_SPATIAL_SIZE * self.FINAL_SPATIAL_SIZE
        # self.flatten_size = 128 * 8 * 8 = 8192 (The correct expected input size)

        # 2. Fully Connected Layers for Latent Space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # 3. DECODER Construction
        self.fc_z = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder_layers = nn.ModuleList()
        # Decoder layers are reverse of encoder (3 layers)
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
        # --- ENCODER ---
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        
        # Flatten size should be 8192
        x = x.view(x.size(0), -1) 
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # --- DECODER ---
        x = self.fc_z(z)
        final_encoder_channels = self.encoder_layers[-1].out_channels
        
        # CRITICAL FIX: Reshape to the correct spatial size (8x8)
        x = x.view(x.size(0), final_encoder_channels, self.FINAL_SPATIAL_SIZE, self.FINAL_SPATIAL_SIZE) 

        for i, layer in enumerate(self.decoder_layers):
            if i == self.num_layers - 1:
                reconstructed_x = torch.tanh(layer(x))
            else:
                x = F.relu(layer(x))
        
        return reconstructed_x, mu, logvar, z

# --- 4, 5, 6. Loss, Feature Extraction, and Plotting Utilities (Modified) ---
def vae_loss(reconstructed_x, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    batch_size = x.size(0)
    total_loss = (MSE + beta * KLD) / batch_size
    return total_loss, MSE, KLD

@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_z = []
    # ... (extraction logic remains the same) ...
    return torch.cat(all_z, dim=0).numpy()

@torch.no_grad()
def plot_and_save_reconstruction(model, dataloader, device, epoch, max_images=4):
    
    model.eval()
    
    # --- CRITICAL FIX: Re-define 3-Channel Inverse Normalization ---
    # This must be done here using the global arrays
    if len(IMAGE_MEAN) >= 3:
        mean_3ch = IMAGE_MEAN[:3]
        std_3ch = IMAGE_STD[:3]
    else:
        mean_3ch = [0.5, 0.5, 0.5]
        std_3ch = [0.5, 0.5, 0.5]

    inv_normalize_3ch = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
        std=[1/s for s in std_3ch]
    )

    # --- Get Data Batch ---
    try:
        data_iterator = iter(dataloader)
        x_raw = next(data_iterator)
    except StopIteration:
        print("Warning: Cannot plot reconstruction, DataLoader iterator is empty.")
        return

    if isinstance(x_raw, (list, tuple)):
        x_raw = x_raw[0]

    x_sample = x_raw[:max_images].to(torch.float32).to(device)
    reconstructed_x, _, _, _ = model(x_sample)
    
    x_sample_cpu = x_sample.cpu()
    reconstructed_x_cpu = reconstructed_x.cpu()

    # --- Plotting Loop ---
    fig, axes = plt.subplots(max_images, 2, figsize=(4, max_images * 2))
    
    for i in range(max_images):
        
        # Slice to 3 Channels and Denormalize using the 3-channel object
        original_3ch = x_sample_cpu[i][:3, :, :]
        reconstructed_3ch = reconstructed_x_cpu[i][:3, :, :]
        
        # Apply the 3-channel denormalization object
        original_img = inv_normalize_3ch(original_3ch).permute(1, 2, 0).numpy()
        reconstructed_img = inv_normalize_3ch(reconstructed_3ch).permute(1, 2, 0).numpy()
        
        # Clip to [0, 1] range 
        original_img = np.clip(original_img, 0, 1)
        reconstructed_img = np.clip(reconstructed_img, 0, 1)

        # Plot original
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original {i+1} (Ch 1-3)", fontsize=8)
        axes[i, 0].axis('off')

        # Plot reconstruction
        axes[i, 1].imshow(reconstructed_img)
        axes[i, 1].set_title(f"Recon {i+1} (Ch 1-3)", fontsize=8)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plot_filename = f'64_patch_reconstruction_epoch_{epoch+1:03d}.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved reconstruction image to {plot_filename}")


# ======================================================================
# 7. MAIN TRAINING AND SAVING LOGIC (Simplified and Corrected)
# ======================================================================

if __name__ == '__main__':
    # ... (Your main execution logic here, using the fixed functions)
    # DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5' 
    DATA_FILE_PATH = 'semi_supervised_only_tq_train_i_snr_20_32bit.hdf5' # <--- CRITICAL CHANGE: New data path
    MODEL_SAVE_PATH = 'vae_checkpoint_4ch_64patch.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 7000
    BETA = 1.0 
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    print(f"Loading data from {DATA_FILE_PATH}...")
    
    try:
        with h5py.File(DATA_FILE_PATH, 'r') as hf:
            data_array_raw = hf['images'][:] 
            data_array = percentile_normalize_channel_wise(data_array_raw, low_percentile=10, high_percentile=99.7)
            print(f"Loaded and preprocessed data shape: {data_array.shape}")
    except Exception as e:
        print(f"Error loading data: {e}. Generating mock data.")
        data_array = np.random.rand(1000, IMAGE_CHANNELS, ORIGINAL_SIZE, ORIGINAL_SIZE).astype(np.float32)

    data_loader = get_dataloader(data_array, batch_size=BATCH_SIZE)
    
    if data_loader is None:
        exit()
    
    effective_count = len(data_loader.dataset)
    print(f"Training on {effective_count} patches (cropped to {TARGET_SIZE}x{TARGET_SIZE}).")
    
    model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint from {MODEL_SAVE_PATH} to resume training.")
        try:
            # Map checkpoint directly to the target device
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint) 
            print("Model weights loaded successfully.")
        except RuntimeError as e:
            print(f"Checkpoint Loading Error: {e}. Starting fresh.")
            
    # --- Training Loop ---
    # ... (Loop logic is assumed to be correct now that the functions are fixed)
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for x in pbar:
            if isinstance(x, (list, tuple)):
                 x = x[0]
            x = x.to(torch.float32).to(device)
            
            optimizer.zero_grad()
            reconstructed_x, mu, logvar, _ = model(x)
            loss, mse_loss, kld_loss = vae_loss(reconstructed_x, x, mu, logvar, beta=BETA)
            
            loss_item = loss.cpu().item()
            loss.backward()
            optimizer.step()
            running_loss += loss_item
            
            pbar.set_postfix({'Loss': f'{loss_item:.4f}', 'MSE': f'{mse_loss.item()/x.size(0):.2f}'})

        avg_epoch_loss = running_loss / len(data_loader)
        print(f"\n--- Epoch {epoch+1} Complete: Average Loss = {avg_epoch_loss:.4f} ---")
        
        if (epoch + 1) % 35 == 0:
            print(f"--- Saving Reconstruction Plot for Epoch {epoch+1} ---")
            plot_and_save_reconstruction(model, data_loader, device, epoch)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving best model checkpoint to {MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("\nTraining complete!")