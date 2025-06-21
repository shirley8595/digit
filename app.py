import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ✅ Generator definition (same as in training script)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, img_dim=28*28):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], 1)
        return self.model(x).view(-1, 1, 28, 28)

# ✅ Helper: One-hot encoding
def one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

# ✅ Load model
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

generator = load_generator()

# ✅ Streamlit UI
st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Select a digit to generate:", list(range(10)))
generate_btn = st.button("Generate Images")

if generate_btn:
    with st.spinner("Generating..."):
        noise_dim = 100
        n_images = 5
        noise = torch.randn(n_images, noise_dim)
        labels = torch.tensor([digit] * n_images)
        labels_onehot = one_hot(labels)

        with torch.no_grad():
            images = generator(noise, labels_onehot)

        # ✅ Plotting
        fig, axs = plt.subplots(1, n_images, figsize=(10, 2))
        for i in range(n_images):
            axs[i].imshow(images[i][0], cmap="gray")
            axs[i].axis("off")
        st.pyplot(fig)
