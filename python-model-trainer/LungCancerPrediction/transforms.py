import torch
import torchvision.transforms.functional as TF

class AffineTransform:
    def __call__(self, image):
        angle = int(torch.randint(low=-25, high=26, size=(1,), dtype=torch.int32).item())   
        xTranslation = int(torch.randint(low=-15, high=16, size=(1,), dtype=torch.int32).item())    
        yTranslation = int(torch.randint(low=-15, high=16, size=(1,), dtype=torch.int32).item())
        xShear = int(torch.randint(low=-15, high=16, size=(1,), dtype=torch.int32).item())
        yShear = int(torch.randint(low=-15, high=16, size=(1,), dtype=torch.int32).item())
        scale = float(torch.rand(1, dtype=torch.float32).item() * 0.2 + 0.9)
        return TF.affine(image, shear=[xShear, yShear], angle=angle, translate=[xTranslation, yTranslation], scale=scale)

