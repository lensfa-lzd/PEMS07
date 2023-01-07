import pandas as pd
import numpy as np
import torch


def make_adj():
    df = pd.read_csv('PEMS07.csv')

    adj = torch.zeros((883, 883))
    start = []
    end = []
    cost = []
    for i in range(df.shape[0]):
        start.append(df.iloc[i, 0])
        end.append(df.iloc[i, 1])
        cost.append(df.iloc[i, 2])

    mean = np.mean(cost)
    std = np.std(cost)

    for i in range(len(start)):
        fr = start[i]
        to = end[i]
        co = np.exp(-(cost[i]**2)/(std**2))

        adj[fr, to] = cost

    for i in range(adj.shape[0]):
        adj[i, i] = 1

    # torch.save(adj, 'adj.pth')


def k_graph():
    adj = torch.load('adj.pth').numpy()
    n_vertex = adj.shape[0]

    graph = np.zeros((n_vertex, n_vertex))
    for i in range(n_vertex):
        n_percent = []
        dis = adj[i]
        for j in range(1, 11):
            n_percent.append(np.percentile(dis, 100-int(j*10)))

        for j in range(10):
            if j == 0:
                top = 1
            else:
                top = n_percent[j-1]
            bottom = n_percent[j]

            for k in range(n_vertex):
                if bottom <= dis[k] <= top:
                    graph[i, k] = j

    # print(graph)
    # graph = torch.from_numpy(graph).float()
    # torch.save(graph, 'k_neighbor.pth')

# data = np.load('PEMS07.npz')['data']
# # (28224, 883, 1)
# print(data.shape)
# torch.save(torch.from_numpy(data).float(), 'vel.pth')
# time_index = pd.date_range('2017-05-01 00:00:00', '2017-08-31 23:55:00', freq='5min')