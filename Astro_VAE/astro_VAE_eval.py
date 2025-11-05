
import torch
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap # Import UMAP library
import matplotlib.pyplot as plt
import os
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

    inv_normalize_3ch = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
        std=[1/s for s in std_3ch]
    )

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
        original_img = inv_normalize_3ch(original_3ch).permute(1, 2, 0).numpy()
        
        # Reconstructed Image (Denormalize and transpose to HWC format)
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
    plot_filename = f'reconstruction_epoch_{epoch+1:03d}.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved reconstruction image to {plot_filename}")

    
# --- 5. Feature Extraction Utility ---
@torch.no_grad()
def extract_features(model, dataloader, device):
    """Passes data through the encoder and collects the latent vectors (z)."""
    model.eval()
    all_z = []
    
    # Use tqdm to show progress during extraction
    pbar = tqdm(dataloader, desc="Extracting Embeddings", unit="batch")
    
    for batch_x in pbar:
        if isinstance(batch_x, (list, tuple)):
             # Ensure we get the tensor itself
             batch_x = batch_x[0] 
        
        # Ensure data is float32 and on the correct device
        x = batch_x.to(torch.float32).to(device)
        
        # Only the latent vector z is needed
        _, _, _, z = model(x)
        
        all_z.append(z.cpu())
        
    # Concatenate all batches and convert to a single NumPy array
    return torch.cat(all_z, dim=0).numpy()

def load_redshifts(file_path):
    try:
        # Assumes the HDF5 file contains a dataset named 'redshifts'
        with h5py.File(file_path, 'r') as hf:
            data_z = hf['redshifts'][:]
            return data_z
    except Exception as e:
        print(f"Error loading redshifts from HDF5: {e}. Returning dummy data.")
        # Return dummy data if file access fails (adjust size as needed)
        return np.random.uniform(0.1, 5.0, 1000)

import torch
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import os
import h5py # Assuming h5py is used to load redshifts

# --- Function to load redshifts (Data source block) ---
def load_redshifts(file_path):
    try:
        # Assumes the HDF5 file contains a dataset named 'redshifts'
        with h5py.File(file_path, 'r') as hf:
            data_z = hf['redshifts'][:]
            return data_z
    except Exception as e:
        print(f"Error loading redshifts from HDF5: {e}. Returning dummy data.")
        # Return dummy data if file access fails (adjust size as needed)
        return np.random.uniform(0.1, 5.0, 1000) 

# --- UMAP Plotting Function ---
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap 

def plot_umap_with_color_encoding(embedding, redshifts, filename='umap_redshift_projection.png'):
    """
    Plots the 2D UMAP embedding, coloring points by their redshift values
    and explicitly setting the color bar range from 0.0 to 2.0.
    
    Args:
        embedding (np.ndarray): The 2D UMAP reduced features.
        redshifts (np.ndarray): The 1D array of redshift values.
        filename (str): The output filename for the plot.
    """
    # Define the desired color limits
    Z_MIN = 0.0
    Z_MAX = 2.0
    
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the 2D representation, setting vmin and vmax
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=redshifts, # Color based on the redshift array
        s=5,         # Point size
        cmap='viridis', 
        alpha=0.7,
        # --- CRITICAL: Set the color limits ---
        vmin=Z_MIN,
        vmax=Z_MAX
    )
    
    plt.gca().set_aspect('equal', 'datalim')
    
    # 2. Add a colorbar to show the constrained redshift scale
    cbar = plt.colorbar(scatter, label=f'Redshift ($z$) [Range {Z_MIN} to {Z_MAX}]')
    
    # Set labels and title
    plt.title('UMAP Projection of VAE Embeddings, Colored by Redshift (Constrained)', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(filename)
    # plt.show()
    plt.close()

# Example Usage (assuming 'embedding' and 'redshift_data' are already defined):
# plot_umap_with_color_encoding(embedding, redshift_data)

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



import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from torchvision import transforms
import torch
import os
from PIL import Image # For image handling
from skimage.transform import resize # For downscaling images cleanly

# Assuming IMAGE_CHANNELS, IMAGE_MEAN, IMAGE_STD, TARGET_SIZE are defined globally
# Also assuming your ConvolutionalVAE, get_dataloader, and extract_features are defined

def plot_umap_with_image_thumbnails(
    embedding: np.ndarray, 
    original_images_tensor: torch.Tensor, # Batch of original 4-channel images (N, C, H, W)
    filename: str = 'umap_image_thumbnails.png',
    num_display_images: int = 1000, # Max number of images to display
    thumbnail_size: int = 32,      # Size of each thumbnail in pixels
    plot_dim: int = 2000,          # Size of the entire output plot in pixels
    jitter_scale: float = 0.5,     # Factor to add slight random jitter to prevent perfect overlap
    seed: int = 42
):
    """
    Plots the 2D UMAP embedding, replacing a subset of points with image thumbnails.

    Args:
        embedding (np.ndarray): The 2D UMAP reduced features (N, 2).
        original_images_tensor (torch.Tensor): The original 4-channel input images (N, 4, 108/128, 108/128).
        filename (str): The output filename for the plot.
        num_display_images (int): Maximum number of images to plot to avoid clutter.
        thumbnail_size (int): Size in pixels for each image thumbnail.
        plot_dim (int): The width and height of the final output image in pixels.
        jitter_scale (float): Scale factor for random jitter.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    
    print(f"Generating UMAP plot with {min(num_display_images, embedding.shape[0])} image thumbnails...")

    # --- 1. Prepare Data and Select Subsample ---
    
    # Scale UMAP coordinates to fit within the plot dimensions
    min_x, max_x = embedding[:, 0].min(), embedding[:, 0].max()
    min_y, max_y = embedding[:, 1].min(), embedding[:, 1].max()

    # Normalize to [0, 1] range, then scale to plot_dim
    scaled_embedding_x = ((embedding[:, 0] - min_x) / (max_x - min_x)) * plot_dim
    scaled_embedding_y = ((embedding[:, 1] - min_y) / (max_y - min_y)) * plot_dim

    # Select a random subset of images to display
    num_total_images = embedding.shape[0]
    if num_total_images > num_display_images:
        display_indices = np.random.choice(num_total_images, num_display_images, replace=False)
    else:
        display_indices = np.arange(num_total_images)

    # Initialize a large blank canvas
    background_image = np.zeros((plot_dim + thumbnail_size, plot_dim + thumbnail_size, 3), dtype=np.uint8) # RGB canvas

    # --- 2. Create Denormalization Transform (for 3 channels) ---
    if len(IMAGE_MEAN) >= 3:
        mean_3ch = IMAGE_MEAN[:3]
        std_3ch = IMAGE_STD[:3]
    else:
        mean_3ch = [0.5, 0.5, 0.5]
        std_3ch = [0.5, 0.5, 0.5]

    # inv_normalize_3ch = transforms.Normalize(
    #     mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
    #     std=[1/s for s in std_3ch]
    # )

    # --- 3. Place Thumbnails on Canvas ---
    for i_idx in display_indices:
        x, y = int(scaled_embedding_x[i_idx]), int(scaled_embedding_y[i_idx])

        # Add slight jitter to avoid perfect overlaps
        x = int(x + np.random.uniform(-jitter_scale, jitter_scale) * thumbnail_size)
        y = int(y + np.random.uniform(-jitter_scale, jitter_scale) * thumbnail_size)

        # Ensure coordinates are within bounds
        x = np.clip(x, 0, plot_dim - thumbnail_size)
        y = np.clip(y, 0, plot_dim - thumbnail_size)

        # Get original image, denormalize first 3 channels, convert to HWC, and resize
        original_tensor_4ch = original_images_tensor[i_idx].cpu() # Ensure on CPU
        original_tensor_3ch = original_tensor_4ch[:3, :, :]       # Take first 3 channels
        
        # Denormalize
        img_np = (original_tensor_3ch).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1) # Clip to [0, 1]

        # Resize for thumbnail
        # Use skimage.transform.resize for cleaner resizing than PIL.Image.resize(array) sometimes
        thumbnail = resize(img_np, (thumbnail_size, thumbnail_size), anti_aliasing=True)
        thumbnail_uint8 = (thumbnail * 255).astype(np.uint8)

        # Place thumbnail on the background image
        background_image[y : y + thumbnail_size, x : x + thumbnail_size] = thumbnail_uint8

    # --- 4. Plotting the Canvas ---
    plt.figure(figsize=(plot_dim/100, plot_dim/100), dpi=100) # Adjust figure size based on plot_dim
    plt.imshow(background_image)
    plt.title('UMAP Projection with Image Thumbnails', fontsize=16)
    plt.xlabel('UMAP Dimension 1 (Scaled)')
    plt.ylabel('UMAP Dimension 2 (Scaled)')
    plt.axis('off') # Turn off axes for a cleaner image mosaic
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()
    print(f"UMAP plot with image thumbnails saved to {filename}")



    
if __name__ == '__main__':
    DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5' # <--- CRITICAL CHANGE: New data path
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
            # min_per_channel = np.min(data_array, axis=(0, 2, 3), keepdims=True)
            # data_array = data_array - min_per_channel
            # data_array = np.log10(1 + data_array)
            # min_per_channel = np.min(data_array, axis=(0, 2, 3), keepdims=True)
            # data_array = data_array - min_per_channel
            # max_per_channel = np.max(data_array, axis=(0, 2, 3), keepdims=True)
            # max_per_channel[max_per_channel == 0] = 1e-10 
            # data_array = data_array / max_per_channel

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
    
    redshift_data = load_redshifts(DATA_FILE_PATH)
    
   
        
        
    print("\n--- Starting Final Feature Extraction for UMAP ---")
    
    # Reload the best saved model
    final_model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
            final_model.load_state_dict(checkpoint)
        except RuntimeError:
            print("Warning: Could not load final checkpoint. Using the last trained epoch model.")
    
    # 1. Extract the latent vectors (embeddings)
    features_Z = extract_features(final_model, data_loader, device)
    print(f"Extracted feature matrix (Z) shape: {features_Z.shape}")

    # 2. Perform UMAP Dimensionality Reduction
    print("Performing UMAP reduction to 2 dimensions...")
    
    # Initialize UMAP reducer (adjust parameters like n_neighbors or min_dist as needed)
    reducer = umap.UMAP(
        n_neighbors=15,          # Controls how local the structure is
        min_dist=0.1,            # Controls how tightly clustered the embeddings are
        n_components=2,          # Target dimensions
        random_state=42          # For reproducible results
    )
    
    # Fit the reducer to the high-dimensional latent features
    embedding = reducer.fit_transform(features_Z)

    print(f"UMAP Embedding shape: {embedding.shape}")
    
     # Ensure the redshift data matches the number of samples in your embedding
    if redshift_data.shape[0] != embedding.shape[0]:
        print("Error: Redshift data size does not match embedding size. Cannot plot.")
    else:
        # --- Step 3: Plot with Color Encoding ---
        plot_umap_with_color_encoding(embedding, redshift_data)
        
    # --- Step 2: Extract Embeddings and Perform UMAP ---
    final_model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(device)
    # ... (Load checkpoint) ...
    features_Z = extract_features(final_model, data_loader, device) # This is your (N, LATENT_DIM) array

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_Z) # This is your (N, 2) array


    # --- Step 3: Call the Image Thumbnail Plotting Function ---
    # CRITICAL: Pass the *full preprocessed data array* (which is now a NumPy array)
    # The plotting function will convert slices of it to torch tensors as needed.
    plot_umap_with_image_thumbnails(
        embedding=embedding, 
        original_images_tensor=torch.from_numpy(data_array), # Convert to torch.Tensor
        num_display_images=200, # Example: plot 200 images
        thumbnail_size=48,      # Example: Make thumbnails 48x48
        plot_dim=1500           # Example: Output plot 1500x1500 pixels
    )

    # You can still call the redshift plotting if you want both visualizations
    # redshift_data = load_redshifts(DATA_FILE_PATH)
    # if redshift_data.shape[0] == embedding.shape[0]:
    #     plot_umap_with_color_encoding(embedding, redshift_data)
    
    