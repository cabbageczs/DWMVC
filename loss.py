import torch
import torch.nn as nn


def cosdis(x1, x2):
    return (1 - torch.cosine_similarity(x1, x2, dim=-1)) / 2


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.t = 1
        self.tripletloss = nn.TripletMarginWithDistanceLoss(margin=1.0, distance_function=cosdis)

    def weighted_wmse_loss(self, input, target, weight, reduction='mean'):
        if isinstance(input, list):
            loss = [0] * len(input)
            for i in range(len(input)):
                loss[i] = torch.mean(weight[:, i:i + 1].mul(target[i] - input[i]) ** 2)
            loss = torch.stack(loss, 0)
        else:
            loss = (weight.unsqueeze(-1).mul(target - input)) ** 2

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        return loss
    
    def selfpaecd_wmse_loss(self, input, target, weight_inc, weights, device, reduction='mean'):
        if isinstance(input, list):
            loss = [0] * len(input)
            for i in range(len(input)):
                weights_expand = weights[:, i:i+1].to(device)
                # loss[i] = torch.mean((weights_expand * weight_inc[:, i:i + 1]).mul(target[i] - input[i]) ** 2)
                loss[i] = torch.mean((weights_expand * (1 - weight_inc[:, i:i + 1]) + weight_inc[:, i:i + 1]).mul(target[i] - input[i]) ** 2)
            loss = torch.stack(loss, 0)
        else:
            weights_expand = weights.unsqueeze(1).to(device)
            loss = ((weights_expand * weight_inc.unsqueeze(-1)).mul(target - input)) ** 2

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        return loss

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f,device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion_selfpaecd = nn.CrossEntropyLoss(reduction="none")
        
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
    
    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
    def forward_feature_selfpaced(self, h_i, h_j, weights):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion_selfpaecd(logits, labels)
        weights_expand = torch.cat((weights, weights), dim=0)
        loss = torch.sum(weights_expand.to(self.device) * loss)
        loss /= N
        return loss
    
    def graph_loss(self, sub_graph, sub_x, all_x):
        if len(sub_graph.shape) == 2:

            diag0_graph0 = torch.diag(sub_graph.sum(1))  # m*m for a m*n matrix
            diag1_graph0 = torch.diag(sub_graph.sum(0))  # n*n for a m*n matrix
            graph_loss = torch.trace(sub_x.t().mm(diag0_graph0).mm(sub_x)) + torch.trace(
                all_x.t().mm(diag1_graph0).mm(all_x)) - 2 * torch.trace(sub_x.t().mm(sub_graph).mm(all_x))
            return graph_loss / (sub_graph.shape[0] * sub_graph.shape[1])
        else:
            graphs_loss = 0
            for v, graph in enumerate(sub_graph):
                diag0_graph0 = torch.diag(graph.sum(1))  # m*m for a m*n matrix
                diag1_graph0 = torch.diag(graph.sum(0))  # n*n for a m*n matrix
                graphs_loss += torch.trace(sub_x[v].t().mm(diag0_graph0).mm(sub_x[v])) + torch.trace(
                    all_x[v].t().mm(diag1_graph0).mm(all_x[v])) - 2 * torch.trace(sub_x[v].t().mm(graph).mm(all_x[v]))
            return graphs_loss / (sub_graph.shape[0] * sub_graph.shape[1] * sub_graph.shape[2])
        