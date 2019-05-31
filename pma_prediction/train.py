import sys
import re
import time
from os.path import isfile
import csv
from sklearn.metrics import r2_score

from model import *
from utils import *

def train(model_path, dataloader, data_size):
    num_epochs = 20
    num_classes = 1 # regression
    num_batches = len(dataloader)
    model = inceptionnet(num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.MSELoss()
    epoch = load_checkpoint(model_path, model) if isfile(model_path) else 0
    filename = re.sub("\.epoch[0-9]+$", "", model_path)
    print("Training model on {} images".format(data_size))
    start_time = time.time()

    for ei in range(epoch + 1, epoch + num_epochs + 1):
        print("Epoch {}/{}".format(ei, num_epochs))
        scheduler.step()
        model.train()
        loss_sum = 0.
        acc_train = 0.
        y_true = []
        y_pred = []
        timer = time.time()
        for i, data in enumerate(dataloader):
            print("\rTraining batch {}/{}".format(i+1, num_batches), end='', flush=True)
            x, y = data['image'], data['age']
            if CUDA:
                x, y = Variable(x.cuda()).float(), Variable(y.cuda()).float()
                model.cuda()
            else:
                x = Variable(x).float(), Variable(y).float()
            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(x)
            out, aux_out = outputs[0], outputs[1]
            out = out.squeeze()
            loss = criterion(out, y)
            # _, preds = torch.max(outputs.data, 1) # for vgg
            # loss = criterion(outputs, y) # for vgg
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            acc_train += torch.sum(abs(out - y.data))
            y_true.extend(y.data.cpu().numpy())
            y_pred.extend(out.detach().cpu().numpy())
            del x, y, outputs
            torch.cuda.empty_cache()

        timer = time.time() - timer
        loss_sum /= data_size
        acc_train /= data_size

        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

        print()
        print("Epoch {} result: ".format(ei))
        print("Avg loss (train): {:.4f}".format(loss_sum))
        print("Avg acc (train): {:.4f}".format(acc_train))
        print("R2 score: {:.4f}".format(r2_score(y_true, y_pred)))

    end_time = time.time()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    train(model_path, dataloader)
