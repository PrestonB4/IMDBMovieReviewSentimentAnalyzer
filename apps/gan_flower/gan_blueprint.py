"""
Flask Blueprint for GAN Flower Generation
"""

from flask import Blueprint, request, jsonify, send_file
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import threading
import time
import psutil
import io
from PIL import Image

gan_bp = Blueprint('gan', __name__, url_prefix='/api/gan')

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Global variables for lazy loading
_generator = None
_device = None
_last_used = None
_cleanup_timer = None
_lock = threading.Lock()

# Cleanup timeout in seconds (10 minutes of inactivity)
CLEANUP_TIMEOUT = 600

# GAN configuration
NZ = 100  # Latent vector dimension
NGF = 64  # Generator filter base size
NC = 3    # RGB channels

class Generator(nn.Module):
    """
    Generator Network: Transforms 100-dim noise vector to 64x64 RGB flower images
    """

    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input layer: Project noise to initial feature map
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer: Upsample
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer: Upsample
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth layer: Upsample
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: Generate final RGB image
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def cleanup_model():
    """Unload model from memory after timeout"""
    global _generator, _device, _cleanup_timer
    with _lock:
        if _generator is not None and time.time() - _last_used > CLEANUP_TIMEOUT:
            print("GAN model inactive for 10 minutes, cleaning up...")
            _generator = None
            _device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("GAN model unloaded from memory")
        _cleanup_timer = None

def schedule_cleanup():
    """Schedule model cleanup after timeout"""
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
    _cleanup_timer = threading.Timer(CLEANUP_TIMEOUT, cleanup_model)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()

def get_generator():
    """Lazy load the trained generator model."""
    global _generator, _device, _last_used

    with _lock:
        if _generator is None:
            # Measure memory before loading
            mem_before = get_memory_usage()
            print(f"Memory before loading GAN: {mem_before:.2f} MB")

            # Set device
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {_device}")

            # Path to model checkpoint
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, 'models', 'gan_final.pth')

            print("Loading GAN generator model...")

            # Initialize generator
            _generator = Generator(nz=NZ, ngf=NGF, nc=NC).to(_device)

            # Load trained weights
            checkpoint = torch.load(model_path, map_location=_device)
            _generator.load_state_dict(checkpoint['generator_state_dict'])
            _generator.eval()  # Set to evaluation mode

            print(f"Loaded GAN generator from {model_path}")

            # Measure memory after loading
            mem_after = get_memory_usage()
            mem_used = mem_after - mem_before
            print(f"Memory after loading GAN: {mem_after:.2f} MB")
            print(f"GAN model uses: {mem_used:.2f} MB")
            print("GAN generator loaded successfully!")

        _last_used = time.time()
        schedule_cleanup()
        return _generator, _device

@gan_bp.route('/generate', methods=['POST'])
def generate_flower():
    """Generate a flower image from random noise."""
    try:
        # Get generator
        generator, device = get_generator()

        # Get optional seed from request (for reproducibility)
        data = request.get_json() or {}
        seed = data.get('seed', None)

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate random noise vector
        noise = torch.randn(1, NZ, 1, 1, device=device)

        # Generate image
        with torch.no_grad():
            fake_image = generator(noise).detach().cpu()

        # Denormalize from [-1, 1] to [0, 1]
        fake_image = fake_image * 0.5 + 0.5
        fake_image = torch.clamp(fake_image, 0, 1)

        # Convert to PIL Image
        fake_image = fake_image.squeeze(0)  # Remove batch dimension
        img_array = fake_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # Save to bytes buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name='generated_flower.png'
        )

    except Exception as e:
        print(f"Error generating flower: {str(e)}")
        return jsonify({'error': str(e)}), 500

@gan_bp.route('/generate-multiple', methods=['POST'])
def generate_multiple_flowers():
    """Generate multiple flower images."""
    try:
        # Get generator
        generator, device = get_generator()

        # Get number of images from request
        data = request.get_json() or {}
        num_images = min(data.get('num_images', 4), 16)  # Max 16 images
        seed = data.get('seed', None)

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate random noise vectors
        noise = torch.randn(num_images, NZ, 1, 1, device=device)

        # Generate images
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()

        # Denormalize from [-1, 1] to [0, 1]
        fake_images = fake_images * 0.5 + 0.5
        fake_images = torch.clamp(fake_images, 0, 1)

        # Convert to base64 images
        images = []
        for i in range(num_images):
            img_tensor = fake_images[i]
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Convert to base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            import base64
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            images.append(f"data:image/png;base64,{img_base64}")

        return jsonify({'images': images})

    except Exception as e:
        print(f"Error generating flowers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@gan_bp.route('/memory', methods=['GET'])
def memory_stats():
    """Get current memory usage statistics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return jsonify({
        'total_memory_mb': mem_info.rss / 1024 / 1024,
        'model_loaded': _generator is not None,
        'message': 'Load the model by calling /generate to see memory increase'
    })
