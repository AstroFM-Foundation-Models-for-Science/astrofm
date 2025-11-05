import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap
import matplotlib.pyplot as plt
import os
import h5py
from skimage.transform import resize

# --- HYPERPARAMETERS (Must match your trained model) ---
LATENT_DIM = 128
IMAGE_CHANNELS = 4
TARGET_SIZE = 64  # Input size used by the VAE architecture
ORIGINAL_SIZE = 108 # Original patch size before cropping
BASE_CHANNELS = 32
IMAGE_MEAN = [0.5] * IMAGE_CHANNELS
IMAGE_STD = [0.5] * IMAGE_CHANNELS

# ======================================================================
# 1. DATA PROCESSING AND LOADING UTILITIES
# ======================================================================

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

def load_redshifts(file_path):
    """Loads redshift data from HDF5 for color encoding."""
    try:
        with h5py.File(file_path, 'r') as hf:
            # Assuming 'redshifts' dataset matches number of images
            return hf['redshifts'][:]
    except Exception as e:
        print(f"Warning: Could not load redshifts from HDF5: {e}. Skipping color coding.")
        return None

class NumPyVAEDataset(Dataset):
    """ Loads data, performs central crop, and applies normalization. """
    def __init__(self, data_array: np.ndarray):
        self.data = data_array.astype(np.float32)
        
        # Only normalization is applied after cropping
        self.transform = transforms.Compose([
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
        
        # Calculate crop indices (64x64 patch from 108x108)
        self.start_index = (ORIGINAL_SIZE - TARGET_SIZE) // 2
        self.end_index = self.start_index + TARGET_SIZE

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_tensor = torch.from_numpy(self.data[idx])
        
        # Perform central crop
        image_tensor_cropped = image_tensor[:, self.start_index:self.end_index, self.start_index:self.end_index]
        
        # Apply normalization
        image_tensor_cropped = self.transform(image_tensor_cropped)
        return image_tensor_cropped

def get_dataloader(data_array: np.ndarray, batch_size: int = 64) -> DataLoader:
    """ Utility function to create and return the DataLoader instance. """
    dataset = NumPyVAEDataset(data_array)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ======================================================================
# 3. VAE Model Definition (MUST MATCH TRAINED MODEL)
# ======================================================================

class ConvolutionalVAE(nn.Module):
    def __init__(self, target_size=TARGET_SIZE, latent_dim=LATENT_DIM, 
                 in_channels=IMAGE_CHANNELS, base_channels=BASE_CHANNELS):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 64 -> 8 spatial size requires 3 layers of stride 2
        self.num_layers = int(np.log2(target_size / 8)) # Should be 3
        self.FINAL_SPATIAL_SIZE = 8 # Final feature map is 8x8
        
        # Channels: 4 -> 32 -> 64 -> 128 (3 layers)
        encoder_channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_layers.append(
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1], kernel_size=4, stride=2, padding=1)
            )
        
        FINAL_CHANNELS = encoder_channels[-1] # 128
        
        # Flatten Size: 128 * 8 * 8 = 8192
        self.flatten_size = FINAL_CHANNELS * self.FINAL_SPATIAL_SIZE * self.FINAL_SPATIAL_SIZE

        # Latent Space Layers (Input size must be 8192)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder Setup (Symmetric)
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

        # --- DECODER (Only needed if reconstruction is desired) ---
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
# 4. EVALUATION UTILITIES
# ======================================================================

@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extracts the latent vector (z) for all data in the DataLoader."""
    model.eval()
    all_z = []
    pbar = tqdm(dataloader, desc="Extracting Embeddings", unit="batch")
    
    for batch_x in pbar:
        if isinstance(batch_x, (list, tuple)):
             batch_x = batch_x[0] 
        
        x = batch_x.to(torch.float32).to(device)
        
        _, _, _, z = model(x)
        all_z.append(z.cpu())
        
    return torch.cat(all_z, dim=0).numpy()

def plot_umap_with_color_encoding(embedding, redshifts, filename='umap_redshift_projection.png'):
    """Plots the UMAP embedding colored by redshift (0 to 2)."""
    Z_MIN = 0.0
    Z_MAX = 2.0
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=redshifts, 
        s=5,         
        cmap='viridis', 
        alpha=0.7,
        vmin=Z_MIN,
        vmax=Z_MAX
    )
    
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(scatter, label=f'Redshift ($z$) [Range {Z_MIN} to {Z_MAX}]')
    plt.title('UMAP Projection of VAE Embeddings, Colored by Redshift', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(filename)
    plt.close()

def plot_umap_with_image_thumbnails(
    embedding: np.ndarray, 
    original_images_tensor: torch.Tensor, 
    filename: str = 'umap_image_thumbnails.png',
    num_display_images: int = 200,
    thumbnail_size: int = 48,
    plot_dim: int = 1500,
    jitter_scale: float = 0.5,
    seed: int = 42
):
    """Plots the UMAP embedding with a subset of images as markers."""
    np.random.seed(seed)
    
    print(f"Generating UMAP plot with {min(num_display_images, embedding.shape[0])} image thumbnails...")

    # Set up scaling
    min_x, max_x = embedding[:, 0].min(), embedding[:, 0].max()
    min_y, max_y = embedding[:, 1].min(), embedding[:, 1].max()
    scaled_embedding_x = ((embedding[:, 0] - min_x) / (max_x - min_x)) * plot_dim
    scaled_embedding_y = ((embedding[:, 1] - min_y) / (max_y - min_y)) * plot_dim

    # Select random subset indices
    num_total_images = embedding.shape[0]
    display_indices = np.random.choice(num_total_images, num_display_images, replace=False)

    # Initialize canvas
    background_image = np.zeros((plot_dim + thumbnail_size, plot_dim + thumbnail_size, 3), dtype=np.uint8) 
    
    # Simple 3-channel normalization for plotting visualization (assuming [0.0, 1.0] scaled data)
    mean_3ch = IMAGE_MEAN[:3]
    std_3ch = IMAGE_STD[:3]
    
    inv_normalize_3ch = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean_3ch, std_3ch)],
        std=[1/s for s in std_3ch]
    )

    # Place Thumbnails
    for i_idx in display_indices:
        x, y = int(scaled_embedding_x[i_idx]), int(scaled_embedding_y[i_idx])

        # Add jitter
        x = np.clip(int(x + np.random.uniform(-jitter_scale, jitter_scale) * thumbnail_size), 0, plot_dim - thumbnail_size)
        y = np.clip(int(y + np.random.uniform(-jitter_scale, jitter_scale) * thumbnail_size), 0, plot_dim - thumbnail_size)

        # Get and preprocess image slice
        original_tensor_4ch = original_images_tensor[i_idx].cpu()
        original_tensor_3ch = original_tensor_4ch[:3, :, :]
        
        img_np = inv_normalize_3ch(original_tensor_3ch).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Resize for thumbnail
        thumbnail = resize(img_np, (thumbnail_size, thumbnail_size), anti_aliasing=True)
        thumbnail_uint8 = (thumbnail * 255).astype(np.uint8)

        # Place thumbnail
        background_image[y : y + thumbnail_size, x : x + thumbnail_size] = thumbnail_uint8

    # Plotting the Canvas
    plt.figure(figsize=(plot_dim/100, plot_dim/100), dpi=100)
    plt.imshow(background_image)
    plt.title('UMAP Projection with Image Thumbnails', fontsize=16)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"UMAP plot with image thumbnails saved to {filename}")


# ======================================================================
# 5. MAIN EXECUTION (EVALUATION LOGIC)
# ======================================================================

if __name__ == '__main__':
    # --- Configuration ---
    # DATA_FILE_PATH = 'semi_supervised_test_i_snr_20.hdf5'
    DATA_FILE_PATH = 'semi_supervised_only_tq_train_i_snr_20_32bit.hdf5'
    
    MODEL_SAVE_PATH = 'vae_checkpoint_4ch_64patch.pth' # Path to your trained model weights
    BATCH_SIZE = 64
    
    # --- Device Selection ---
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print(f"Loading and preprocessing data from {DATA_FILE_PATH}...")
    
    try:
        with h5py.File(DATA_FILE_PATH, 'r') as hf:
            data_array_raw = hf['images'][:] 
            redshift_data_full = hf['redshifts'][:] # Load redshifts too
            
            # 1. Perform channel-wise percentile normalization
            data_array = percentile_normalize_channel_wise(data_array_raw, low_percentile=10, high_percentile=99.7)
            print(f"Preprocessed data shape: {data_array.shape}")
            
    except Exception as e:
        print(f"FATAL ERROR: Could not load HDF5 data or redshifts: {e}.")
        raise SystemExit(1)

    # Create DataLoader (shuffle=False for consistent results)
    data_loader = get_dataloader(data_array, batch_size=BATCH_SIZE)
    
    # --- Model Setup and Loading ---
    print(f"\nLoading model from {MODEL_SAVE_PATH}...")
    final_model = ConvolutionalVAE(target_size=TARGET_SIZE, latent_dim=LATENT_DIM).to(torch.float32).to(device)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Checkpoint file not found at {MODEL_SAVE_PATH}")
        raise SystemExit(1)
        
    try:
        # Load weights and map to device
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
        final_model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Error loading model weights (architecture mismatch): {e}")
        raise SystemExit(1)

    # ======================================================================
    # RUN EVALUATION AND PLOTTING
    # ======================================================================
    
    # 1. Extract Embeddings
    features_Z = extract_features(final_model, data_loader, device)
    print(f"Extracted latent features shape: {features_Z.shape}")

    # 2. Perform UMAP Dimensionality Reduction
    print("Performing UMAP reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_Z)
    print(f"UMAP Embedding shape: {embedding.shape}")

    # 3. Plot UMAP with Redshift Color Encoding
    if redshift_data_full.shape[0] == embedding.shape[0]:
        plot_umap_with_color_encoding(embedding, redshift_data_full, filename='umap_redshift_colored.png')
    else:
        print("Warning: Redshift data size mismatch. Skipping redshift plot.")

    # 4. Plot UMAP with Image Thumbnails
    plot_umap_with_image_thumbnails(
        embedding=embedding, 
        original_images_tensor=torch.from_numpy(data_array), # Use preprocessed data
        num_display_images=300, 
        thumbnail_size=48,      
        plot_dim=1500
    )
    
    print("\nEvaluation complete. Plots generated.")