import os
import torch
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms, models

from networks import FaceBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset

import visdom
import numpy as np

def train():
    use_gpu = torch.cuda.is_available()
    file_root = os.path.dirname(os.path.abspath(__file__))

    learning_rate = 0.001
    num_epochs = 300
    batch_size = 32

    net = FaceBox()
    if use_gpu:
        net.cuda()

    print('load model...')
    net.load_state_dict(torch.load('weight/faceboxes.pt'))


    criterion = MultiBoxLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[198, 248], gamma=0.1)

    train_dataset = ListDataset(root=file_root,
                                list_file='data/train_rewrite.txt',
                                train=True,
                                transform = [transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = ListDataset(root=file_root,
                                list_file='data/val_rewrite.txt',
                                train=False,
                                transform = [transforms.ToTensor()])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))

    num_iter = 0
    vis = visdom.Visdom()
    win = vis.line(Y=np.array([0]), X = np.array([0]))

    net.train()
    for epoch in range(num_epochs):
        scheduler.step()

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.
        net.train()
        for i,(images,loc_targets,conf_targets) in enumerate(train_loader):

            if use_gpu:
                images = images.cuda()
                loc_targets = loc_targets.cuda()
                conf_targets = conf_targets.cuda()

            loc_preds, conf_preds = net(images)
            loss = criterion(loc_preds,loc_targets,conf_preds,conf_targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))

                vis.line(Y=np.array([total_loss / (i+1)]), X=np.array([num_iter]),
                        win=win,
                        name='train',
                        update='append')
                num_iter += 1
        val_loss = 0.0
        net.eval()
        for idx, (images, loc_targets,conf_targets) in enumerate(val_loader):
            with torch.no_grad():
                if use_gpu:
                    images = images.cuda()
                    loc_targets = loc_targets.cuda()
                    conf_targets = conf_targets.cuda()

                loc_preds, conf_preds = net(images)
                loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
                val_loss += loss.item()
        val_loss /= len(val_dataset)/batch_size
        vis.line(Y=np.array([val_loss]), X=np.array([epoch*40+40]),
                 win=win,
                 name='val',
                 update='append')
        print('loss of val is {}'.format(val_loss))

        if not os.path.exists('weight/'):
            os.mkdir('weight')

        print('saving model ...')
        torch.save(net.state_dict(),'weight/faceboxes.pt')
    

if __name__ == '__main__':
    train()