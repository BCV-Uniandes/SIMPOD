import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Utils import seed_everything
from Load_data import SG_Dataloaders
from Architectures import model_selector
from torch.optim.lr_scheduler import ReduceLROnPlateau


def Train(dataloader, epoch, Cuda, optimizer, model, loss, fold = 1):
    total_loss = 0
    if Cuda: 
        model.cuda()
    model.train()
    dataloader_size = len(dataloader)
    dataloader = tqdm(enumerate(dataloader))
    for idx, (x, SG) in dataloader:
        if Cuda: 
            x, SG = x.cuda(), SG.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss_ = loss(output, SG)
        loss_.backward()
        optimizer.step()
        total_loss += loss_.cpu().item()
        dataloader.set_description(f'Fold:{fold} Epoch:{epoch} Epoch percentage:{round((idx/dataloader_size)*100, 2)}%  loss: {round(loss_.cpu().item(), 4)}')
    avg_loss = total_loss/dataloader_size
    return avg_loss


def Test(dataloader, Cuda, model, loss):
    correct = 0
    top3_correct = 0
    top5_correct = 0
    total_loss = 0
    if Cuda: 
        model.cuda()
    model.eval()
    size = len(dataloader)
    size2 = len(dataloader.dataset)
    dataloader = tqdm(enumerate(dataloader))
    for idx, (x, SG) in dataloader:
        if Cuda: 
            x, SG = x.cuda(), SG.cuda()
        with torch.no_grad():
            output = model(x)  
            total_loss += loss(output, SG).cpu().item()
            pred = output.data.max(1)[1]
            top3_pred = torch.topk(output.data, 3, 1)[1]
            top5_pred = torch.topk(output.data, 5, 1)[1]
            top3_SG = SG.unsqueeze(1).expand(-1, 3)
            top5_SG = SG.unsqueeze(1).expand(-1, 5)
            correct += pred.eq(SG.data).cpu().sum().item()
            top3_correct += top3_pred.eq(top3_SG.data).cpu().sum().item()
            top5_correct += top5_pred.eq(top5_SG.data).cpu().sum().item()
            dataloader.set_description(f'Data percentage:{round((idx/size)*100, 2)}%')
    avg_loss = total_loss/size
    accuracy = 100 * correct/size2
    top3_accuracy = 100 * top3_correct /size2
    top5_accuracy = 100 * top5_correct/size2
    print('Average loss: ', avg_loss)
    print('Accuracy: ', accuracy)
    print('Top 3 Accuracy:', top3_accuracy)
    print('Top 5 Accuracy:', top5_accuracy)
    return avg_loss, accuracy, top3_accuracy, top5_accuracy


def Total_train(epochs, dataloaders, Cuda, model, optimizer, scheduler, name, fold = 1):
    train_dataloader, valid_dataloader = dataloaders
    best_accuracy = None
    loss = nn.CrossEntropyLoss()
    for i in range(epochs):
        print('-'*20,f'Training Fold {fold}...', '-'*20)
        train_loss = Train(train_dataloader, i+1, Cuda, optimizer, model, loss, fold = fold)
        print('-'*20,f'Valid Results Fold {fold}', '-'*20)
        _, _, _, top5_accuracy = Test(valid_dataloader, Cuda, model, loss) 
        scheduler.step(train_loss)
        if best_accuracy is None or top5_accuracy > best_accuracy:
            best_accuracy = top5_accuracy
            with open(os.path.join('Space_Group_Prediction','Models' , name + '_Best.pt'), 'wb') as fp:
                state = model.state_dict()
                torch.save(state, fp)
    with open(os.path.join('Space_Group_Prediction', 'Models', name + '_Last.pt'), 'wb') as fp:
        state = model.state_dict()
        torch.save(state, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'alexnet')
    parser.add_argument('--pretrained', type = str, default = 'True')
    parser.add_argument('--epochs', type = int, default = 25)
    parser.add_argument('--training_data', type = str, default= '100K')
    parser.add_argument('--batch_size', type = int, default= 32)
    parser.add_argument('--lr', type = float, default = 0.00001)
    parser.add_argument('--gamma', type = float, default = 0.9)
    parser.add_argument('--patience', type = float, default = 2)
    args = parser.parse_args()
    seed_everything(7)
    path_file = open(os.path.join('Space_Group_Prediction', 'Paths', 'paths.json'))
    paths_data = json.load(path_file)[0]
    paths_data = paths_data[args.training_data]
    fold1 = paths_data['Fold1']
    fold2 = paths_data['Fold2']
    SG_images_folder = os.path.join('Data', 'Powder_images')
    dataloaders1 = SG_Dataloaders(args.batch_size, fold1, fold2, SG_images_folder)
    dataloaders2 = SG_Dataloaders(args.batch_size, fold2, fold1, SG_images_folder)
    Cuda = True
    Pretrained = False if args.pretrained == 'False' else True
    model1 = model_selector(args.model, pretrained = Pretrained)
    model2 = model_selector(args.model, pretrained = Pretrained)
    optimizer1 = optim.Adam(model1.parameters(), lr = args.lr, betas = (0.9, 0.999))
    optimizer2 = optim.Adam(model2.parameters(), lr = args.lr, betas = (0.9, 0.999))
    scheduler1 = ReduceLROnPlateau(optimizer1, factor = args.gamma, patience = args.patience, mode = 'min')
    scheduler2 = ReduceLROnPlateau(optimizer2, factor = args.gamma, patience = args.patience, mode = 'min')
    epochs = args.epochs
    name1 = args.model + '_pretrained_' + args.pretrained + '_lr_' + str(round(args.lr, 6)) + '_bs_' + str(args.batch_size) + '_epochs_' + str(epochs) + '_gamma_' + str(round(args.gamma, 4)) + '_patience_' + str(int(args.patience)) + '_data_' + str(args.training_data) + '_fold1'
    name2 = args.model + '_pretrained_' + args.pretrained + '_lr_' + str(round(args.lr, 6)) + '_bs_' + str(args.batch_size) + '_epochs_' + str(epochs) + '_gamma_' + str(round(args.gamma, 4)) + '_patience_' + str(int(args.patience)) + '_data_' + str(args.training_data) + '_fold2'
    Total_train(epochs, dataloaders1, Cuda, model1, optimizer1, scheduler1, name1, fold = 1)
    Total_train(epochs, dataloaders2, Cuda, model2, optimizer2, scheduler2, name2, fold = 2)


if __name__ == '__main__':
    main()
    