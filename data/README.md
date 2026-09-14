\# Gnasher Group Data Plan



\## Dataset



\*\*Dataset:\*\* Stanford Dogs Dataset



\*\*Source:\*\* Stanford University



\*\*Dataset Link:\*\* http://vision.stanford.edu/aditya86/ImageNetDogs/



The Stanford Dogs Dataset contains \*\*20,580 dog images representing 120 breeds\*\*. The original breed labels will be mapped into the seven American Kennel Club (AKC) breed groups for this project.



\## Classification Labels



The model will classify dog images into the following seven AKC breed groups:



\- Sporting

\- Hound

\- Working

\- Terrier

\- Toy

\- Non-Sporting

\- Herding



\## Planned Dataset Size



The full Stanford Dogs Dataset contains 20,580 images. For this project, approximately \*\*7,000 balanced images\*\* will be used, with a maximum of about \*\*1,000 images per AKC group\*\*.



This smaller balanced dataset will help reduce class imbalance while keeping training practical using the free Google Colab GPU environment.



\## Data Split



The dataset will be divided into:



\- \*\*70% Training\*\*

\- \*\*15% Validation\*\*

\- \*\*15% Testing\*\*



\## Data Organization



```text

data/

├── raw/       # Original dog images from the Stanford Dogs Dataset

└── processed/ # Images organized and labeled into the seven AKC groups

```



\## Data Preparation



The original Stanford Dogs breed labels will be mapped to their corresponding AKC breed groups. Images will then be prepared for use with the EfficientNet-B0 image-classification model using PyTorch and Torchvision.



\## License / Compliance



The Stanford Dogs Dataset will be used for educational and academic purposes. The project will follow the dataset's stated usage terms and properly acknowledge the original dataset source.

