import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random 
import h5py # New import for loading data efficiently

# --- HYPERPARAMETERS ---
LATENT_DIM = 256
IMAGE_CHANNELS = 4
TARGET_SIZE = 128 # <--- CRITICAL CHANGE: New input size (128x128)
CROP_SIZE_FOR_RESIZE = 128 # Use 128 for simplicity
BASE_CHANNELS = 32
IMAGE_MEAN = [0.5, 0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5, 0.5]

# ======================================================================
# 1. WSI UTILITY CLASSES (REMOVED/REPLACED)
# ======================================================================
# The WSI reading logic is removed as data is now loaded from a NumPy/HDF5 file.

# ======================================================================
# 2. VAE NUMPY ARRAY DATASET (New Custom PyTorch Dataset)
# ======================================================================

class NumPyVAEDataset(Dataset):
    """ Loads data directly from a NumPy array (or HDF5 dataset). """
    def __init__(self, data_array: np.ndarray):
        # Data shape is assumed to be (N, C, H, W) -> (N, 4, 108, 108)
        self.data = data_array
        self.data = data_array.astype(np.float32)

        # --- TRANSFORMATION PIPELINE: RESIZE (108 -> 128) -> NORMALIZE ---
        transform_steps = [
            # CRITICAL ADDITION: Resize the 108x108 tensor to 128x128
            transforms.Resize((TARGET_SIZE, TARGET_SIZE), 
                              transforms.InterpolationMode.BILINEAR)
        ]
            
        transform_steps.extend([
            # Removed ToTensor() because input is already a float32 tensor
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        self.transform = transforms.Compose(transform_steps)
        
        # Ensure data is correctly shaped (N, C, H, W) and float32
        if self.data.ndim == 4:
             # Ensure channels are first (C, H, W) and data is float32
             self.data = self.data.astype(np.float32)
        else:
             raise ValueError(f"Input data must be 4D (N, C, H, W), but shape is {self.data.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        # Get the tensor for a single item (C, H, W)
        image_tensor = torch.from_numpy(self.data[idx])
        
        # Apply normalization transform
        image_tensor = self.transform(image_tensor)
        return image_tensor

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

# --- 3. VAE Model Definition (CRITICAL FIXES APPLIED) ---
class ConvolutionalVAE(nn.Module):
    def __init__(self, target_size=TARGET_SIZE, latent_dim=LATENT_DIM, 
                 in_channels=IMAGE_CHANNELS, base_channels=BASE_CHANNELS):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # In ConvolutionalVAE.__init__:
        # Calculate number of downsampling steps needed to reach 8x8 spatial size
        # 128 / 8 = 16. log2(16) = 4 layers.
        self.num_layers = int(np.log2(target_size / 8)) # Will be 4

        if 2**self.num_layers * 8 != target_size:
            raise ValueError(...) # This check will now pass for 128
        

        # CRITICAL FIX: The spatial size after 4 layers of stride 2 conv on 108x108 input

        # ... (rest of ENCODER and DECODER construction remains the same)
        # Assuming the dynamic list construction still correctly yields [4, 32, 64, 128, 256]
        encoder_channels = [in_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        self.encoder_layers = nn.ModuleList()
        # Layers for 108x108 input (4 layers of stride 2)
        for i in range(self.num_layers):
            self.encoder_layers.append(
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=4, stride=2, padding=1)
            )
        
        self.FINAL_SPATIAL_SIZE = encoder_channels[-1]
        FINAL_CHANNELS = encoder_channels[-1]
        # Final spatial size is correctly 8x8
        self.flatten_size = FINAL_CHANNELS * 8 * 8

        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

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
        # --- ENCODER ---
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        
        x = x.view(x.size(0), -1) 
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # --- DECODER ---
        x = self.fc_z(z)
        final_encoder_channels = self.encoder_layers[-1].out_channels
        
        SPATIAL_SIZE_AFTER_ENCODER = 8 

        x = x.view(x.size(0), final_encoder_channels, SPATIAL_SIZE_AFTER_ENCODER, SPATIAL_SIZE_AFTER_ENCODER)

        
        for i, layer in enumerate(self.decoder_layers):
            if i == self.num_layers - 1:
                reconstructed_x = torch.tanh(layer(x))
            else:
                x = F.relu(layer(x))
        
        return reconstructed_x, mu, logvar, z

# --- 4, 5, 6. Loss, Feature Extraction, and Plotting Utilities (No major changes) ---
def vae_loss(reconstructed_x, x, mu, logvar, beta=1.0):
    # ... (Loss function remains the same)
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    batch_size = x.size(0)
    total_loss = (MSE + beta * KLD) / batch_size
    return total_loss, MSE, KLD

@torch.no_grad()
def extract_features(model, dataloader, device):
    # ... (Feature extraction remains the same)
    model.eval()
    all_z = []
    for batch_x in dataloader:
        if isinstance(batch_x, (list, tuple)):
             batch_x = batch_x[0] 
        x = batch_x.to(device)
        _, _, _, z = model(x)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0).numpy()

@torch.no_grad()
def plot_and_save_reconstruction(model, dataloader, device, epoch, max_images=4):
    """
    Saves a comparison figure of a batch of original images vs. their reconstructions.
    
    CRITICAL FIX: Uses a 3-channel inverse normalization object for plotting
    to match the sliced tensor ([:3, :, :]), resolving the RuntimeError.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms # Import locally for utility functions

    model.eval()
    
    # --- 1. Define the 3-Channel Inverse Normalization Object ---
    
    # Safely get the first 3 channels' mean and std for visualization
    # This prevents the RuntimeError by ensuring the transform matches the tensor size (3)
    if len(IMAGE_MEAN) >= 3:
        mean_3ch = IMAGE_MEAN[:3]
        std_3ch = IMAGE_STD[:3]
    else:
        # Fallback for error handling if configuration is severely wrong
        mean_3ch = [0.5, 0.5, 0.5]
        std_3ch = [0.5, 0.5, 0.5]

    # inv_normalize_3ch = transforms.Normalize(
    #     mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
    #     std=[1/s for s in std_3ch]
    # )

    # --- 2. Get Data Batch ---
    try:
        data_iterator = iter(dataloader)
        x_raw = next(data_iterator)
    except StopIteration:
        print("Warning: Cannot plot reconstruction, DataLoader iterator is empty.")
        return
    except Exception as e:
        print(f"Warning: Error getting batch for plot: {e}")
        return

    if isinstance(x_raw, (list, tuple)):
        x_raw = x_raw[0]

    # Process the first few images only (ensure they are float32 before moving)
    x_sample = x_raw[:max_images].to(torch.float32).to(device)

    # --- 3. Pass through VAE ---
    reconstructed_x, _, _, _ = model(x_sample)
    
    # Move tensors to CPU for plotting
    x_sample_cpu = x_sample.cpu()
    reconstructed_x_cpu = reconstructed_x.cpu()

    # --- 4. Plotting Loop ---
    fig, axes = plt.subplots(max_images, 2, figsize=(4, max_images * 2))
    
    for i in range(max_images):
        
        # --- Slice to 3 Channels and Denormalize using the 3-channel object ---
        original_3ch = x_sample_cpu[i][:3, :, :]
        reconstructed_3ch = reconstructed_x_cpu[i][:3, :, :]
        
        # Original Image (Denormalize and transpose to HWC format)
        # original_img = inv_normalize_3ch(original_3ch).permute(1, 2, 0).numpy()
        original_img = (original_3ch).permute(1, 2, 0).numpy()
        
        # Reconstructed Image (Denormalize and transpose to HWC format)
        # reconstructed_img = inv_normalize_3ch(reconstructed_3ch).permute(1, 2, 0).numpy()
        reconstructed_img = (reconstructed_3ch).permute(1, 2, 0).numpy()
        
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
    plot_filename = f'reconstruction_epoch_{epoch+1:03d}.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved reconstruction image to {plot_filename}")

# Note: This function requires the global variables IMAGE_MEAN, IMAGE_STD, and 
# IMAGE_CHANNELS to be defined for the 4-channel configuration.


# ======================================================================
# 7. MAIN TRAINING AND SAVING LOGIC
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


if __name__ == '__main__':
    # --- Configuration ---
    # DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5' # <--- CRITICAL CHANGE: New data path
    DATA_FILE_PATH = 'semi_supervised_only_tq_train_i_snr_20_32bit.hdf5' # <--- CRITICAL CHANGE: New data path
    
    MODEL_SAVE_PATH = 'vae_checkpoint_4ch.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 250
    BETA = 1.0 
    
    # --- Device Selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # --- Data Loading (From NumPy Array/HDF5) ---
    print(f"Loading data from {DATA_FILE_PATH}...")
    
    # 1. Load data into a NumPy array (Modify this block for your HDF5 file)
    try:
        # Assuming you store your data in a single HDF5 dataset named 'images'
        with h5py.File(DATA_FILE_PATH, 'r') as hf:
             # Load the array into memory. Shape must be (N, 4, 108, 108)
            data_array_raw = hf['images'][:] 

            # data_array = data_array_raw[:, 1:5, :, :].astype(np.float32)
            data_array=percentile_normalize_channel_wise(data_array_raw, low_percentile=10, high_percentile=99.7)
            
            # data_array = data_array_raw[:, 1:5, :, :].astype(np.float32)
            # min_per_channel = np.min(data_array, axis=(1,2,3), keepdims=True)
            # max_per_channel = np.max(data_array, axis=(1,2,3), keepdims=True)
            # data_array = data_array - min_per_channel
            # data_array = data_array / (max_per_channel - min_per_channel + 1e-10)
            # # data_array=data_array**0.25
            # data_array = np.log(1 + data_array)
            # min_per_channel = np.min(data_array, axis=(1,2,3), keepdims=True)
            # data_array = data_array - min_per_channel
            # max_per_channel = np.max(data_array, axis=(1,2,3), keepdims=True)
            # max_per_channel[max_per_channel == 0] = 1e-10 
            # data_array = data_array / max_per_channel
            # data_array=data_array**2

            #  data_array = data_array_raw[:,1:5,:,:].astype(np.float32)  # Ensure float32 type
             
            #  data_array=data_array-np.min(data_array)
            #  data_array=np.log10(1+data_array)
            #  data_array=data_array-np.min(data_array)
            #  data_array=data_array/np.max(data_array)
            print(f"Loaded data shape: {data_array.shape}")
    except Exception as e:
        print(f"Error loading data from HDF5: {e}. Generating mock data.")
        # Fallback to generating a mock array
        data_array = np.random.rand(1000, IMAGE_CHANNELS, TARGET_SIZE, TARGET_SIZE).astype(np.float32)

    data_loader = get_dataloader(data_array, batch_size=BATCH_SIZE)
    
    if data_loader is None:
        print("Exiting due to empty dataset.")
        exit()
    
    effective_count = len(data_loader.dataset)

    print(f"Training with an effective dataset size of {effective_count}.")
    print(f"Input Shape: ({IMAGE_CHANNELS}, {TARGET_SIZE}, {TARGET_SIZE})")
    
    # --- Model Setup ---
    # Ensure model is float32 for initial setup compatibility before moving to MPS
    model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Checkpoint Loading and Continuation Logic ---
    # ... (The checkpoint logic remains the same, but loading model to float32 first)
    best_loss = float('inf')
    start_epoch = 0
    
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
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for batch_x in pbar:
            if isinstance(batch_x, (list, tuple)):
                 x = batch_x[0]
            else:
                 x = batch_x

            # CRITICAL FIX: Ensure both model and input are float32 on the correct device
            x = x.to(torch.float32).to(device)
            
            optimizer.zero_grad()
            reconstructed_x, mu, logvar, _ = model(x)
            loss, mse_loss, kld_loss = vae_loss(reconstructed_x, x, mu, logvar, beta=BETA)
            
            # Move loss back to CPU only for item() if necessary (less error-prone)
            loss_item = loss.cpu().item()
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss_item
            
            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}', 
                'MSE': f'{mse_loss.item()/x.size(0):.2f}', 
                'KLD': f'{kld_loss.item()/x.size(0):.2f}'
            })

        # ... (rest of loop logic for saving and plotting)
        avg_epoch_loss = running_loss / len(data_loader)
        print(f"\n--- Epoch {epoch+1} Complete: Average Loss = {avg_epoch_loss:.4f} ---")
        plot_and_save_reconstruction(model, data_loader, device, epoch)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving best model checkpoint to {MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("\nTraining complete!")
    print(f"Best VAE model saved to {MODEL_SAVE_PATH}")