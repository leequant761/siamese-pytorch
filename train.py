from absl import flags
import sys
from collections import deque
import os
import pickle
import time

from mydataset import OmniglotTrain, OmniglotTest
from model import Siamese

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

def train(model, train_loader, optimizer):
    model.train()
    for batch_id, (img1, img2, label) in enumerate(train_loader, 1):
        # if batch_id > Flags.max_iter:
        if batch_id > 10:
            break
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output = model.forward(img1, img2)
        loss = F.binary_cross_entropy_with_logits(output, label, reduction='mean')
        loss.backward()
        optimizer.step()
    
    return loss.item()

def evaluate(model, test_loader):
    model.eval()
    right, error = 0, 0
    with torch.no_grad():
        for _, (test1, test2) in enumerate(test_loader, 1):
            test1, test2, label = test1.to(DEVICE), test2.to(DEVICE)
            output = model.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1 # test1's class == test2[0]'s class
            else:
                error += 1 # test1's class != test2[other]'s class 
    test_accuracy = right*1.0/(right+error)
    return test_accuracy

if __name__ == '__main__':

    Flags = flags.FLAGS
    flags.DEFINE_bool("cuda", True, "use cuda")
    flags.DEFINE_string("train_path", "C:/Users/lee/Desktop/leequant761/siamese-pytorch/omniglot/python/images_background", "training folder")
    flags.DEFINE_string("test_path", "C:/Users/lee/Desktop/leequant761/siamese-pytorch/omniglot/python/images_evaluation", 'path of testing folder')
    flags.DEFINE_integer("way", 20, "how much way one-shot learning")
    flags.DEFINE_integer("times", 400, "number of samples to test accuracy")
    flags.DEFINE_integer("workers", 1, "number of dataLoader workers")
    flags.DEFINE_integer("batch_size", 128, "number of batch size")
    flags.DEFINE_integer("num_batches", 10, "number of batch size")
    flags.DEFINE_float("lr", 0.00006, "learning rate")
    flags.DEFINE_integer("save_every", 10, "save model after each save_every iter.")
    flags.DEFINE_integer("Epochs", 5000, "number of iterations before stopping")
    flags.DEFINE_string("model_path", "C:/Users/lee/Desktop/leequant761/siamese-pytorch/models", "path to store model")
    flags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    # os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    # print("use gpu:", Flags.gpu_ids, "to train.")

    trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
    testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times = Flags.times, way = Flags.way)
    
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    net = Siamese()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)
    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr = Flags.lr)

    for epoch in range(1, Flags.Epochs + 1):
        time_start = time.time()
        train_loss = train(net, trainLoader, optimizer)
        test_accuracy = evaluate(net, testLoader)
        print('[%d / %d] loss: %.5f | acc: %f | time_lapsed: %.2f sec'%(epoch, 50000, train_loss, test_accuracy, time.time() - time_start))
        if epoch % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(epoch) + ".pt")