import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image


#args = parse_args()

CUDA_DEVICES = torch.cuda.is_available()
DATASET_ROOT = 'C:/Users/allen/Desktop/Kaggle-Plant-Seedlings-Classification-Example-master'
#PATH_TO_WEIGHTS = args.weight
PATH_TO_WEIGHTS = 'C:/Users/allen/Desktop/Kaggle-Plant-Seedlings-Classification-Example-master/model-0.87-best_train_acc.pth'

def test():

    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_root = Path(DATASET_ROOT)
    classes = [_dir.name for _dir in dataset_root.joinpath('train').glob('*')]

    model = torch.load(PATH_TO_WEIGHTS)

    if CUDA_DEVICES:
        model = model.cuda(1)
    model.eval()

    sample_submission = pd.read_csv(str(dataset_root.joinpath('sample_submission.csv')))

    submission = sample_submission.copy()

    for i, filename in enumerate(sample_submission['file']):
        image = Image.open(str(dataset_root.joinpath('test').joinpath(filename))).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        if CUDA_DEVICES:
            inputs = Variable(image.cuda(1))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        submission['species'][i] = classes[preds[0]]

    submission.to_csv(str(dataset_root.joinpath('submission.csv')), index=False)


if __name__ == '__main__':
    test()
