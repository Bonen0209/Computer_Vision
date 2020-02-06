import sys
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ResNet
from data import get_dataloader_vgg
from helper import plot_loss_accuracy

plot_path = 'HW2/hw2_result/hw2-4_result/problem2/'

def main():
    folder = sys.argv[1]

    train_loader, val_loader = get_dataloader_vgg(folder, batch_size=32, num_workers=cpu_count())

    model = ResNet(layers=[2, 2, 2, 2])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    y_trains = [[], []]
    y_valids = [[], []]

    ep = 20
    for epoch in range(ep):
        print('Epoch:', epoch+1)
        
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        for batch, (x, label) in enumerate(train_loader, 1):
            optimizer.zero_grad()

            if use_cuda:
                x, label = x.cuda(), label.cuda()

            out = model(x)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            if batch % 50 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print (f'Training batch index: {batch}, train loss: {ave_loss:.6f}, acc: {acc:.3f}', sep='\t')
        
        y_trains[0].append(correct_cnt / total_cnt)
        y_trains[1].append(total_loss / batch)


        model.eval()
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        for batch, (x, label) in enumerate(val_loader,1):
            if use_cuda:
                x, label = x.cuda(), label.cuda()
                
            out = model(x)
            
            loss = criterion(out, label)

            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

        acc = correct_cnt / total_cnt
        ave_loss = total_loss / batch           
        print ('Validating batch index: {}, valid loss: {:.6f}, acc: {:.3f}'.format(batch, ave_loss, acc))

        y_valids[0].append(correct_cnt / total_cnt)
        y_valids[1].append(total_loss / batch)

        model.train()

    torch.save(model.state_dict(), './HW2/checkpoint/%s.pth' % model.name())

    x_epoch = list(range(ep))

    plot_loss_accuracy(x_epoch, y_trains, plot_path, f'{model.name()}_train')
    plot_loss_accuracy(x_epoch, y_valids, plot_path, f'{model.name()}_valid')

if __name__ == "__main__":
    main()