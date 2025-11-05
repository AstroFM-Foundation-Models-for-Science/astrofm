# AstroFM

This repo was developed during the "Foundation Models for Science" workshop at the University of Toronto in November 2025.

We used an astronomy foundation model (AION) and other non-astro foundation models to obtain embeddings of galaxy images. These galaxies are from a new dataset observed by a new telescope. None of the models have been trained on this data. We then used these embeddings to predict distances. We obtained uncertainties on the outputs using MC dropout. Surprisingly, the performance of one non-astro foundation model (ViT) was similar to AION. We fine-tuned the AION model using LORA to obtain even better distance estimations (reduced bias by two orders of magnitude).

<img width="761" height="738" alt="umap_images" src="https://github.com/user-attachments/assets/4558e978-0fce-481b-a333-8b7c7e3385d3" />

These are predicted distances (redshifts, z) plotted against the true distances. 

<img width="559" height="356" alt="Screenshot 2025-11-05 at 3 14 10 PM" src="https://github.com/user-attachments/assets/0d8fd251-c41a-4f19-b6d3-ea13135c568c" />

