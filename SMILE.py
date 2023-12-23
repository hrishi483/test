import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

#Model Class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = self.pool3(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@st.cache_data 
def load_model():
  model=torch.load("D:\AI Adventures\Deep Learning\AlexNet_smile_high_accuracy.pth",map_location=device)
  return model

def predict_image(model,img):
    #preprocessing
    model.eval()
    # image_path = "/content/Mom.jpg"
    # new_img = Image.open(image_path)
    new_img = img
    trans = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor()])
    with torch.no_grad():
        process_img = trans(new_img)
        process_img = process_img.unsqueeze(0)
        process_img = process_img.to(device)
        _,pred = torch.max(model(process_img),1)
    title="Smiling"
    if pred ==0:
        title="Not Smiling"
        return title,_
    else:
        return title,_
    

# model = load_model()
# print(model.parameters)
st.title("Detect SMILING Face ")


image = st.file_uploader(label="Insert Face Image")
# image = st.camera_input(label="Take a picture")
if image:
    model = load_model()
    # st.write(model.parameters)
    pil_image = Image.open(image)
    pil_image = pil_image.convert("RGB")
    prediction,out = predict_image(model,pil_image)
    st.write(out)
    st.success(f"Probably You are {prediction}")
    st.image(image)
    


