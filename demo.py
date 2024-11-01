import os
import io
import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, request, send_file
from PIL import Image
import random
app = Flask(__name__)

# Define the model classes (Conv, Downs, Ups, UNET) here...

# Initialize the model and load the weights

import torchvision.transforms as transforms


# Define the model classes
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class Downs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class Ups(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = Conv(out_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Downs(1, 64)  # Change input channels from 3 to 1
        self.down2 = Downs(64, 128)
        self.down3 = Downs(128, 256)
        self.down4 = Downs(256, 512)

        self.bottleneck = Conv(512, 1024)

        self.up1 = Ups(1024, 512)
        self.up2 = Ups(512, 256)
        self.up3 = Ups(256, 128)
        self.up4 = Ups(128, 64)

        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        skip1, down1 = self.down1(inputs)
        skip2, down2 = self.down2(down1)
        skip3, down3 = self.down3(down2)
        skip4, down4 = self.down4(down3)

        b = self.bottleneck(down4)

        up1 = self.up1(b, skip4)
        up2 = self.up2(up1, skip3)
        up3 = self.up3(up2, skip2)
        up4 = self.up4(up3, skip1)

        outputs = self.outputs(up4)

        return outputs

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

# Initialize the model and load the weights
set_seed(171)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET().to(device)
model.load_state_dict(torch.load( r"c:\Users\NavjeetHundal\Downloads\checkpoint(1).pth", map_location=device))
model.eval()


def preprocess_image(file_stream):
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image


def postprocess_output(output):
    output = output[0].cpu().numpy()
    output = np.squeeze(output, axis=0)
    output = output > 0.5
    output = (output * 255).astype(np.uint8)
    return output


def mask_to_image(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image_stream = file.stream

    # Preprocess the image
    image_tensor = preprocess_image(image_stream).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)

    binary_mask = postprocess_output(output)
    binary_mask_image = mask_to_image(binary_mask)

    # Save to a BytesIO object
    mask_image = Image.fromarray(binary_mask_image)
    img_io = io.BytesIO()
    mask_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
