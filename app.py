# ==============================================================================
# Problem 3: Handwritten Digit Image Generator - Streamlit App
# ==============================================================================
# This script creates a web application that loads the pre-trained cGAN generator
# and generates 5 images of a user-selected digit.

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


st.set_page_config(layout="wide")
# --- MODEL AND APP CONFIGURATION ---

# These parameters MUST match the parameters used during training in the Colab notebook.
LATENT_DIM = 100
N_CLASSES = 10
EMBEDDING_DIM = 100
IMG_SHAPE = (1, 28, 28)
MODEL_PATH = "cgan_generator_epoch_60.pth" # Path to your downloaded model file

# Set device to CPU, as Streamlit Cloud runs on CPU instances.
device = torch.device("cpu")

# --- PYTORCH MODEL DEFINITION ---

# The Generator class definition must be included here so that PyTorch knows
# the architecture of the model when loading the saved weights.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(N_CLASSES, EMBEDDING_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBEDDING_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(IMG_SHAPE))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *IMG_SHAPE)
        return img

# --- HELPER FUNCTION TO LOAD THE MODEL ---

# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_model():
    """Loads the pre-trained generator model."""
    generator = Generator().to(device)
    # Load the state dictionary, ensuring it's mapped to the CPU
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval() # Set the model to evaluation mode
    return generator

generator = load_model()

# --- STREAMLIT USER INTERFACE ---


st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a trained Conditional GAN model.")

# --- USER INPUT AND IMAGE GENERATION ---

st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.sidebar.button("Generate Images"):
    st.subheader(f"Generated images of digit {selected_digit}")

    with st.spinner(f"Generating 5 images of the digit '{selected_digit}'..."):
        # Prepare inputs for the generator
        num_images = 5
        # Generate 5 random noise vectors
        z = torch.randn(num_images, LATENT_DIM, device=device)
        # Create a tensor of the selected digit label, repeated 5 times
        labels = torch.LongTensor([selected_digit] * num_images).to(device)

        # Generate images using the model
        with torch.no_grad():
            generated_images = generator(z, labels)

        # Post-process images for display
        # Denormalize from [-1, 1] to [0, 1]
        generated_images = 0.5 * (generated_images + 1)
        # Squeeze the channel dimension for grayscale display
        generated_images = generated_images.squeeze(1).cpu().numpy()

        # Display the 5 generated images in 5 columns
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            with col:
                # Convert numpy array to an image
                image_array = (generated_images[i] * 255).astype(np.uint8)
                st.image(image_array, caption=f"Sample {i+1}", use_column_width=True)
else:
    st.info("Choose a digit from the sidebar and click 'Generate Images' to start.")