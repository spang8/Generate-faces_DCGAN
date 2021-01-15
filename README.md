# Generate-faces_DCGAN

## Goal
- Train deep convolutional generative adversarial network (DCGAN) on 200,000 human faces to generate new image of faces that look as realistic as possible. 
- The generated faces are fairly realistic with small amounts of noise 

--------------------------------------------


## Procedures 
#### 1. Data preprocess
- Image transform 
    - Crop images to remove parts of the image that don't include a face
    - Resize down to 64x64x3 numpy image
    - ToTensor
- Create dataloader 
    - Define batch size and image size to 32x32x3
    - Shuffle images
- Scaling: scale data from [0,1] to [-1,1]

#### 2. DCGAN architecture  
DCGAN contains a discriminator network and a generator network
- **Discriminator(D)** (conv layers become deeper)
    - image input --> conv1 --> leaky relu -->
    - conv2 --> batch norm --> leaky relu -->
    - conv3 --> batch norm --> leaky relu -->
    - fc ouput layer
- **Generator(G)** (conv layers become thinner: transpose conv)
    - latent vector z as input --> fc layer -->
    - t_conv1 --> batch norm --> relu -->
    - t_conv2 --> batch norm --> relu -->
    - t_conv3 --> tanh
- Initialize weight in normal distribution to help converge
- **A complete network = D + G**

#### 3. Train the model
- Loss functions
    - real image loss (pred = D_out, true = 1 or 0.9): calc in D.
    - fake image loss (pred = D_out, true = 0): calc in D and G
- Optimizer
- **Discriminator training**
    - scale real images --> forward pass real images --> d_loss_real
    - generate fake images --> forward pass fake images --> d_loss_fake
    - d_loss = d_loss_real + d_loss_fake
    - backpropagate
- **Generator training**
    - generate fake images --> Forward pass fake images --> g_loss_fake (use real image loss function - flipped labels)
    - backpropagate
    
#### 4. Generate human faces based on trained model 


--------------------------------------------


## How to improve model performance
#### 1. Diverse the training image datasets. 
- The results are biased as the training images are mostly white.
- Add more images, and add diverse races of faces. 

#### 2. Build a deeper model
- Increase conv layers in discriminator and generator. layer Depth increased

#### 3. Tunning hyperparameters
- Tune learning rate, and beta1. Use grit of parameters. [Ref](https://cs231n.github.io/neural-networks-3/).
    - For learning rate: 10**np.random.uniform(-3,-4) [0.001,0.0001]
    - For beta1: 10**np.random.uniform(0,-1) [0,0.1] 
    - Based on the papers [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), the best parameters for the mentioned paper was lr = 0.0002 and beta1=0.5.
    
- learning rate decay method 

- Increase the number of epoches (would help less as the loss had little change after around 50 epoches)   
