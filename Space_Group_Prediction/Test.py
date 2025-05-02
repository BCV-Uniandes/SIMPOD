import os
import json
import torch
import argparse
from tqdm import tqdm
from Load_data import SG_Dataloaders
from Architectures import model_selector


def Test(dataloader, Cuda, model1, model2):
    correct = 0
    top3_correct = 0
    top5_correct = 0
    if Cuda: 
        model1.cuda()
        model2.cuda()
    model1.eval()
    model2.eval()
    size = len(dataloader)
    size2 = len(dataloader.dataset)
    dataloader = tqdm(enumerate(dataloader))
    for idx, (x, SG) in dataloader:
        if Cuda: 
            x, SG = x.cuda(), SG.cuda()
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
            final_output = (output1 + output2)/2
            pred = final_output.data.max(1)[1]
            top3_pred = torch.topk(final_output.data, 3, 1)[1]
            top5_pred = torch.topk(final_output.data, 5, 1)[1]
            top3_SG = SG.unsqueeze(1).expand(-1, 3)
            top5_SG = SG.unsqueeze(1).expand(-1, 5)
            correct += pred.eq(SG.data).cpu().sum().item()
            top3_correct += top3_pred.eq(top3_SG.data).cpu().sum().item()
            top5_correct += top5_pred.eq(top5_SG.data).cpu().sum().item()
            dataloader.set_description(f'Data percentage:{round((idx/size)*100, 2)}%')
    accuracy = 100 * correct/size2
    top3_accuracy = 100 * top3_correct /size2
    top5_accuracy = 100 * top5_correct/size2
    print('Accuracy: ', accuracy)
    print('Top 3 Accuracy:', top3_accuracy)
    print('Top 5 Accuracy:', top5_accuracy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'alexnet')
    parser.add_argument('--weights1', type = str, default='alexnet_pretrained_True_lr_1.8e-05_bs_12_epochs_25_gamma_0.6043_patience_3_data_100K_fold1_Best.pt')
    parser.add_argument('--weights2', type = str, default='alexnet_pretrained_True_lr_1.8e-05_bs_12_epochs_25_gamma_0.6043_patience_3_data_100K_fold2_Best.pt')
    parser.add_argument("--batch_size", type = int, default = 32)
    args = parser.parse_args()
    paths_file = os.path.join('Space_Group_Prediction', 'Paths', 'paths.json')
    test_paths = json.load(open(paths_file))[0]['All']['Test'][-25000:]
    SG_images_folder = os.path.join('Data', 'Powder_images')
    test_dataloader = SG_Dataloaders(args.batch_size, None, None, SG_images_folder,  json_paths_test=test_paths)
    Cuda = True
    model1 = model_selector(args.model)
    model1.load_state_dict(torch.load(os.path.join('Space_Group_Prediction', 'Models', args.weights1)))
    model2 = model_selector(args.model)
    model2.load_state_dict(torch.load(os.path.join('Space_Group_Prediction', 'Models', args.weights2)))
    Test(test_dataloader, Cuda, model1, model2)

if __name__ == '__main__':
    main()
