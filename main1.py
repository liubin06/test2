import argparse
import os
import pandas
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
import random

import utils
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))
print('main1notsampled1')

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def cdf_trans(cdf_u,alpha,tau_plus):
    tau_minus = 1-tau_plus
    a = (1-2*alpha)*(tau_minus-tau_plus)+1e-8
    b = 2*(alpha*tau_minus + (1-alpha)*tau_plus)
    return (-b + torch.sqrt(b**2 + 4*a*cdf_u))/(2*a)


def criterion(out_1, out_2, tau_plus, batch_size,temperature, alpha, beta, estimator):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        scores = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).to(device)
        neg = scores.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        if estimator=='SimCLR':
            Ng = neg.sum(dim=-1)

        elif estimator == 'DCL':
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))

        elif estimator=='HCL':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))

        elif estimator == 'BCL':
            hatx = neg.unsqueeze(-1)
            X = scores.unsqueeze(1)
            #idx = torch.arange(0, scores.shape[0])
            #X = torch.cat([scores, scores[idx - 1, :],scores[idx - 2, :],scores[idx - 3, :]], dim=1).unsqueeze(1)
            cdf_u = (X <= hatx).sum(dim=-1) / X.shape[-1]
            cdf = cdf_trans(cdf_u,alpha,tau_plus)
            ccdf = 1 - cdf
            phi1 = 2 * ccdf
            phi2 = 2 * cdf

            x_tn = alpha*phi1 + (1-alpha)* phi2
            x_fn = (1-alpha)*phi1 + alpha* phi2

            x_htn = (alpha * (1 - beta) * phi1 + (1 - alpha) * beta * phi2) / (alpha * (1 - beta) + (1 - alpha) * beta+1e-8)
            omega = x_htn / (x_tn * (1-tau_plus) + x_fn*tau_plus)
            Ng = (omega * neg).sum(dim=-1)

        else:
            raise Exception('Invalid estimator selected. Please use any of [SimCLR,DCL,HCL,BCL]')
            # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss

def train(net, data_loader, train_optimizer, temperature, estimator, tau_plus,alpha, beta):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = criterion(out_1, out_2, tau_plus, batch_size,temperature, alpha, beta, estimator)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    macro_auc, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # sample one bach data for estimating alpha
        sample = random.sample(range(len(memory_data_loader)),1)[0]
        counter = 0
        for data, _, target in memory_data_loader:
            if counter == sample:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                # feature [test_batch=256, D维度]
                feature, out = net(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [test_batch=256, N=50000]
                sim_matrix = torch.mm(feature, feature_bank)
                # label[test_batch=256, N=50000]
                bool_labels = target.unsqueeze(-1) == feature_labels
                print('calculating AUC')
                auc = roc_auc_score(bool_labels.T.cpu(), sim_matrix.T.cpu(), average='macro')
            counter += 1
    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='BCL', type=str, help='Choose loss function: SimCLR, DCL, HCL, BCL')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose dataset')
    parser.add_argument('--alpha', default=0.5, type=float, help='Location Parameter valued in [0.5,1]')
    parser.add_argument('--beta', default=1.0, type=float, help='Concentration Parameter valued in [0,Infinity]')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')

    # args parse
    args = parser.parse_args()

    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs,  args.estimator
    dataset_name = args.dataset_name
    alpha = args.alpha
    beta = args.beta
    anneal = args.anneal

    #configuring an adaptive beta if using annealing method
    if anneal=='down':
        do_beta_anneal=True
        n_steps=10
        betas=iter(np.linspace(beta,0.7,n_steps))
        #print([i for i in betas])
    else:
        do_beta_anneal=False
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    if not os.path.exists('../results/{}'.format(dataset_name)):
        os.makedirs('../results/{}'.format(dataset_name))

    for epoch in range(1, epochs + 1):
        # if (epoch-1) % 20 == 0:
        #     alpha = test(model, memory_loader, test_loader)
        # if (epoch-1) % 40 == 0:
        #     beta = next(betas)
        print(alpha,beta)
        train_loss = train(model, train_loader, optimizer, temperature, estimator, tau_plus, alpha, beta)

        if epoch >= 300:
            if epoch % 20 == 0:
                #动态alpha 动态beta
                torch.save(model.state_dict(),'../results/{}/{}_{}_model_{}_{}_{}_{}.pth'.format(dataset_name, dataset_name, estimator,batch_size, alpha, beta, epoch))
