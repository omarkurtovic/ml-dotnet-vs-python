1. A difference was when exporting the images after augmentation.
In c# it was very difficult since it doesn't have a default encoding for images, 
while in python it was very easy to export the images after augmentation.


2. Very different for showing the parameters of the models.
In python we have a nice little library called torchinfo which shows us all the layers, their output parameters and more interesting stuff.
In c# the best we can do is get the tensors of the parameters so the weights and biases using named_parameters() method.
We can at least get the layers that have some parameters like the convolutional and fully connected and can check that the number of parameters
is the same in both ecosystems.


3. Had to adjust logic for augmenting images since torch sharp does not have RandomAffine
Inside pytorch we can create a transform pipeline with torchvision.transforms.RandomAffine which will translate, shear and scale our images 
automatically. But since torchsharp does not have this functionality to keep things consistent we had to implement a custom AffineTransform in both 
ecosystems that creates random numbers for the augmentations and then calls affine manually from torchvision.transforms.functional