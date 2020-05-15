from torchvision import models
import torch



model = models.vgg16(pretrained=True)
print(model.features)
example = torch.rand(1,3,224,224)

model.eval()
script = torch.jit.trace(model,example)
torch.jit.save(script, "vgg16.pt")


