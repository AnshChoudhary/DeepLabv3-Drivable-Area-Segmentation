import argparse
import torch
from dataloaders import make_data_loader
from modeling.deeplab import *
from utils.metrics import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='bdd')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)  # Added batch-size argument
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, num_classes = make_data_loader(args, **{'train': False, 'val': False, 'test': True})
    
    model = DeepLab(num_classes=num_classes, backbone=args.backbone).to(device)
    
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    evaluator = Evaluator(num_classes)
    
    for i, sample in enumerate(test_loader):
        image, target = sample['image'].to(device), sample['label'].to(device)
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    fwIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print(f"Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {fwIoU}")

if __name__ == "__main__":
    main()
