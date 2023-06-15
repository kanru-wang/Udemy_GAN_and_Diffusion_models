# WGAN

https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

In practice, GAN can optimize the discriminator easier than the generator. Assuming the discriminator is already optimal, when minimizing the GAN objective function, if two distributions are very distant (i.e. if the generated image has distribution far away from the ground truth), the divergence would be large, but the gradient of the divergency would eventually diminish. We would have a close-to-zero gradient, i.e. the generator learns nothing from the gradient descent.

<img src="https://miro.medium.com/1*-VajV2qCbPWDCdNGbQfCng.png" width="400"/>

For GAN (the red line), it fills with areas with diminishing or exploding gradients.

For WGAN (the blue line), the gradient is smoother everywhere and learns better even the generator is not producing good images.

Two significant contributions for WGAN are
- it has no sign of mode collapse in experiments, and
- the generator can still learn when the critic (discriminator) performs well.

# WGAN-GP

https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

Wasserstein GAN requires enforcing the Lipschitz constraint, but GAN’s weight clipping has issues. Instead of weight clipping, WGAN-GP uses gradient penalty to enforce the Lipschitz constraint. Gradient penalty penalizes the model if the gradient norm moves away from its target norm value 1.

Batch normalization is avoided for the WGAN-GP’s critic (discriminator). Batch normalization creates correlations between samples in the same batch. It impacts the effectiveness of the gradient penalty.

WGAN-GP demonstrates better image quality and convergence comparing with WGAN. However, DCGAN demonstrates slightly better image quality and it converges faster. Compared to DCGAN, WGAN-GP’s advantage is that its training convergency is more stable, allowing us to use a more complex model like a deep ResNet for the generator and the discriminator.

# Progressive Growing GAN (ProGAN)

https://machinelearningmastery.com/introduction-to-progressive-growing-generative-adversarial-networks/

https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2

<img src="https://machinelearningmastery.com/wp-content/uploads/2019/06/Example-of-Progressively-Adding-Layers-to-Generator-and-Discriminator-Models.png" width="500"/>

<img src="https://machinelearningmastery.com/wp-content/uploads/2019/06/Example-of-Phasing-in-the-Addition-of-New-Layers-to-the-Generator-and-Discriminator-Models.png" width="600"/>

<img src="https://miro.medium.com/v2/1*lStHChxfyLB3S7wUW3Quiw.png" width="700"/>

（Also check out the Discriminator layer diagram from reference link)

- Apart from the high resolution, ProGAN can be trained about 2–6 times faster than a corresponding traditional GAN.
- The phasing in of a new block of layers involves using a skip connection to connect the new block to the input of the discriminator or output of the generator, and adding it to the existing input or output layer with a weighting. The weighting controls the influence of the new block and is achieved using a parameter alpha (α) that starts at zero (or a very small number), and linearly increases to 1.0 over training iterations.
- All layers remain trainable during the training process, including existing layers when new layers are added.
- The upsampling method is nearest neighbor interpolation, which is different from many GAN generators that use a transpose convolutional layer. The downsampling method is average pooling, which is different from many GAN discriminators that use a 2×2 stride in the convolutional layers to downsample.
- Batch normalization is not used; instead, minibatch standard deviation and pixel-wise normalization are used.
    - Minibatch standard deviation: The standard deviation of activations across images in the mini-batch is added as a new channel prior to the last block of convolutional layers in the discriminator. This encourages the generator to produce more variety, such that statistics computed across a generated batch more closely resemble those from a training data batch.
    - Pixel-wise normalization: Normalizes the feature vector in each pixel (of all channels) to unit length, and is applied after the convolutional layers in the generator. This is to prevent signal magnitudes going out of control.
- Equalized Learning Rate: Before every forward pass during training, scale the weights of a layer according to how many weights that layer has.
- Image generation uses a weighted average of prior models rather a given model snapshot, much like a horizontal ensemble.)
- The paper uses WGAN-GP as the loss function, but can use other loss functions.


