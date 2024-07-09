import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from modeling.deeplab import DeepLab

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DeepLab(num_classes=2, backbone="resnet").to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Transform for input images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load input images
    input_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for input_file in input_files:
            image = Image.open(input_file).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)[0]

            # Save prediction
            pred_image = Image.fromarray(pred.astype(np.uint8))
            pred_image.save(os.path.join(args.output, os.path.basename(input_file)))

if __name__ == "__main__":
    main()

