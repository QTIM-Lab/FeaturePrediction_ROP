import sys
import re
import time
import csv
from sklearn.metrics import r2_score

from model import *
from utils import *

def evaluate(model_path, dataloader, data_size):
    num_batches = len(dataloader)
    criterion = nn.MSELoss()
    num_classes = 1
    model = inceptionnet(num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.train(False)
    model.eval()
    print("Evaluating model on {} images".format(data_size))
    start_time = time.time()

    loss_test = 0.
    acc_test = 0.
    y_true = []
    y_pred = []

    csv_file_name = model_path.split('/')[-1] + '_' + str(data_size) + '_predictions.csv'
    with open(csv_file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image', 'plus', 'age', 'predicted age'])

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("\rEvaluating batch {}/{}".format(i+1, num_batches), end='', flush=True)
                x, y, plus, imgs = data['image'], data['age'], data['plus'], data['img']
                if CUDA:
                    x, y = Variable(x.cuda()).float(), Variable(y.cuda()).float()
                    model.cuda()
                else:
                    x = Variable(x).float(), Variable(y).float()
                predicted = model(x)
                predicted = predicted.squeeze()
                loss = criterion(predicted, y)
                loss_test += loss.item()
                acc_test += torch.sum(abs(predicted - y.data))
                y_true.extend(y.data.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                for i in range(len(imgs)):
                    writer.writerow([imgs[i], plus[i].item(), y.data[i].item(), predicted[i].item()])

                del x, y, predicted
                torch.cuda.empty_cache()

    loss_test /= data_size
    acc_test /= data_size
    elapsed_time = time.time() - start_time
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(loss_test))
    print("Avg acc (test): {:.4f}".format(acc_test))
    print("R2 score: {:.4f}".format(r2_score(y_true, y_pred)))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    evaluate(model_path, dataloader)