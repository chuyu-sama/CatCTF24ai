import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
from Crypto.Util.number import long_to_bytes

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img) 
        output = self.fc(feature.view(img.shape[0], -1))  
        return output  


transform = transforms.Compose([
    transforms.ToTensor(),  
    #transforms.Normalize((0.5,), (0.5,)) 就是被归一化坑了。。。
])


model = LeNet()
model.load_state_dict(torch.load('lenet_model.pth', map_location=torch.device('cpu')))

model.eval()

flag_dir = './flag'

image_files = sorted(os.listdir(flag_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

flag_digits = []

with torch.no_grad():  
    for image_file in image_files:
        image_path = os.path.join(flag_dir, image_file)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0) 
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  
        
        print(f"Image: {image_file}, Outputs: {outputs}, Predicted: {predicted.item()}")

        flag_digits.append(str(predicted.item()))

flag = ''.join(flag_digits)
print(f'finaloutput:{flag}') #最后long to bytes一下
flag=long_to_bytes(flag)
print(flag)
