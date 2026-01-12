import torch
import torch.nn as nn
from math import sqrt

torch.cuda.empty_cache()


class SimAttention(nn.Module):  # 通道注意力
    def __init__(self, num, feature, viewNum):
        super(SimAttention, self).__init__()

        self.Num = num
        self.FeaSize = feature
        self.viewn = viewNum
        self.dropout = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(self.viewn, 150 * self.viewn, bias=False)
        self.fc_2 = nn.Linear(150 * self.viewn, self.viewn, bias=False)
        self.GAP1 = nn.AvgPool2d((self.Num, self.Num),
                                 (1, 1))  # 这是一个全局平均池化（GAP）层，会将每个view的num × num相似度矩阵压缩为一个单值（即：通道级统计）

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, similarity):
        avr_pool = self.GAP1(similarity)

        sim_atten = avr_pool.reshape(-1, avr_pool.size(0))

        sim_atten = self.fc_1(sim_atten)
        sim_atten = self.relu(sim_atten)
        sim_atten = self.fc_2(sim_atten)

        sim_atten = sim_atten.reshape(similarity.size(0), 1, 1)
        all_att = self.softmax(sim_atten)
        sim = all_att * similarity

        final_sim = torch.sum(sim, dim=0, keepdim=False)

        return final_sim


# 用于将三种不同类型的边关系（m-m、d-d、m-d）的相似度分数转化为 离散的类别标签矩阵
class OnehotTran(nn.Module):
    def __init__(self, sim_class, md_class, m_num, d_num):
        super(OnehotTran, self).__init__()

        self.m_class = sim_class  # m-m 和 d-d 的相似度划分的类别数（通常为 3）
        self.d_class = sim_class  # m-d 边的类别数（也是 3，值：-1, 1, 2）
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.M_num = m_num
        self.D_num = d_num
        # Define tensors for the classification of associated edges.
        self.m_one = torch.ones(self.M_num, self.M_num)
        self.d_one = torch.ones(self.D_num, self.D_num)
        self.md_one = torch.ones(self.M_num, self.D_num)
        self.mone = self.m_one
        self.mtwo = self.mone + 1
        self.mthr = self.mtwo + 1
        self.done = self.d_one + self.d_class
        self.dtwo = self.done + 1
        self.dthr = self.dtwo + 1
        self.mdone = self.md_one + (self.md_class * 2)
        self.mdtwo = self.mdone + 1
        self.mdthr = self.mdtwo + 1

    def forward(self, m_score, d_score, md_score):
        # edge classification
        mnew_score = torch.where(torch.ge(m_score, 0.65), self.mthr.cuda(), m_score)
        mnew_score = torch.where(torch.ge(mnew_score, 0.35) & torch.lt(mnew_score, 0.65), self.mtwo.cuda(), mnew_score)
        mnew_score = torch.where(torch.gt(mnew_score, 0.0) & torch.lt(mnew_score, 0.35), self.mone.cuda(), mnew_score)

        dnew_score = torch.where(torch.ge(d_score, 0.65), self.dthr.cuda(), d_score)
        dnew_score = torch.where(torch.ge(dnew_score, 0.35) & torch.lt(dnew_score, 0.65), self.dtwo.cuda(), dnew_score)
        dnew_score = torch.where(torch.gt(dnew_score, 0.0) & torch.lt(dnew_score, 0.35), self.done.cuda(), dnew_score)

        mdnew_score = torch.where(torch.eq(md_score, -1.0), self.mdone.cuda(), md_score)
        mdnew_score = torch.where(torch.eq(mdnew_score, 1.0), self.mdtwo.cuda(), mdnew_score)
        mdnew_score = torch.where(torch.eq(mdnew_score, 2.0), self.mdthr.cuda(), mdnew_score)

        pre_one = torch.cat((mnew_score, mdnew_score), dim=1)
        pre_two = torch.cat((mdnew_score.t(), dnew_score), dim=1)
        md_onepre = torch.cat((pre_one, pre_two), dim=0)

        return md_onepre


class NodeEmbedding(nn.Module):
    def __init__(self, m_num, d_num, feature, dropout):
        super(NodeEmbedding, self).__init__()

        self.m_Num = m_num
        self.d_Num = d_num
        self.node_voca_num = m_num + d_num
        self.fea_Size = feature
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.node_voca_num, self.fea_Size)
        self.relu = nn.ReLU()

    def forward(self, m_sim, d_sim, nei_node_list):
        # Define the node feature matrix.
        batch_size = nei_node_list[0].size(0)
        md_f = torch.zeros(self.m_Num, self.d_Num).cuda()
        prep_m_f = torch.cat((m_sim, md_f), dim=1)
        prep_d_f = torch.cat((md_f.t(), d_sim), dim=1)
        m_d_f = torch.cat((prep_m_f, prep_d_f), dim=0)
        md_node_fea = self.drop(self.relu(self.linear(m_d_f)))

        neinode_emb_list = []
        for index in range(len(nei_node_list)):
            nei_node = nei_node_list[index]
            nei_node = torch.reshape(nei_node, (-1, 1)).squeeze(dim=1)
            nei_node_emb = torch.index_select(md_node_fea, 0, nei_node)
            neinode_emb_list.append(torch.reshape(nei_node_emb, (batch_size, -1, self.fea_Size)))

        return neinode_emb_list


class EdgeEmbedding(nn.Module):
    def __init__(self, sim_class, md_class, nei_size):
        super(EdgeEmbedding, self).__init__()

        self.m_class = sim_class
        self.d_class = sim_class
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.neigh_size = nei_size
        self.bottom = torch.arange(start=0, end=self.class_all, step=1)
        self.bottom_onehot = torch.nn.functional.one_hot(self.bottom, self.class_all).float()

    def forward(self, nei_rel_list):
        # Define the edge feature matrix.
        batch_size = nei_rel_list[0].size(0)
        one_hot_emb = self.bottom_onehot.cuda()

        neirel_emb_list = []
        for index in range(len(nei_rel_list)):
            nei_relation = nei_rel_list[index]
            nei_relation = torch.reshape(nei_relation, (-1, 1)).squeeze(dim=1)
            nei_rel_emb = torch.index_select(one_hot_emb, 0, nei_relation)
            neirel_emb_list.append(torch.reshape(nei_rel_emb, (batch_size, -1, self.class_all)))

        return neirel_emb_list


class NeiAttention(nn.Module):
    def __init__(self, edgeFea, nodeFea, nei_size):
        super(NeiAttention, self).__init__()

        self.neigh_size = nei_size
        self.norm = 1 / sqrt(nodeFea)
        self.W1 = nn.Linear(edgeFea + nodeFea, nodeFea)
        self.actfun = nn.Softmax(dim=-1)

    def forward(self, x, x_nei_rel, x_nei_node, i):
        # It mainly implements node-level attention.
        now_nei_size = self.neigh_size[i]
        n_neibor = int(int(x_nei_node.shape[1]) / now_nei_size)
        x = x.unsqueeze(dim=2)
        x_nei = torch.cat((x_nei_rel, x_nei_node), dim=-1)
        x_nei_up = self.W1(x_nei)
        x_nei_val = torch.reshape(x_nei_up, (x.shape[0], n_neibor, now_nei_size, -1))
        alpha = torch.matmul(x, x_nei_val.permute(0, 1, 3, 2)) * self.norm
        alpha = self.actfun(alpha)
        alpha = alpha.permute(0, 1, 3, 2)
        out = alpha * x_nei_val
        outputs = torch.sum(out, dim=2, keepdim=False)

        return outputs


class NeiAggregator(nn.Module):
    def __init__(self, nodeFea, dropout, actFunc, outBn=False, outAct=True, outDp=True):
        super(NeiAggregator, self).__init__()

        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(nodeFea)
        self.out = nn.Linear(nodeFea * 2, nodeFea)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x, x_nei):

        x_new = torch.cat((x, x_nei), dim=-1)
        x_new = self.out(x_new)

        if self.outBn:
            if len(x_new.shape) == 3:
                x_new = self.bns(x_new.transpose(1, 2)).transpose(1, 2)
            else:
                x_new = self.bns(x_new)
        if self.outAct: x_new = self.actFunc(x_new)
        if self.outDp: x_new = self.dropout(x_new)

        return x_new


class Attention(nn.Module):
    def __init__(self, edgeinSize, NodeinSize, outSize):
        super(Attention, self).__init__()

        self.edgeInSize = edgeinSize
        self.NodeInsize = NodeinSize
        self.outSize = outSize
        self.q = nn.Linear(self.edgeInSize, outSize)
        self.k = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.v = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=-1)

    def forward(self, query, input):
        # It mainly implements hyperedge-level attention.
        Q = self.q(query)
        K = self.k(input)
        V = self.v(input)
        alpha = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm
        alpha = self.actfun1(alpha)
        z = (alpha.permute(0, 2, 1)) * V
        outputs = torch.sum(z, dim=1, keepdim=False)

        return outputs


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=False, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)
        if self.outBn: x = self.bns(x)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x
