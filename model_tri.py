import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import networkx as nx
from otherlayers import *
from extractSubGraph import GetSubgraph
from hypergraph import construct_hypergraph, build_multi_type_hypergraphs, generate_G_from_H
from attention import Multi_view_Attention
from kan import *


class SimMatrix(nn.Module):
    def __init__(self, param):
        super(SimMatrix, self).__init__()
        self.mnum = param.m_num
        self.dnum = param.d_num
        self.viewn = param.view
        self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
        self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)

    def forward(self, data):
        m_funsim = data['mm_f'].cuda()
        m_seqsim = data['mm_s'].cuda()
        m_gossim = data['mm_g'].cuda()
        d_funsim = data['dd_t'].cuda()
        d_semsim = data['dd_s'].cuda()
        d_gossim = data['dd_g'].cuda()

        m_sim = torch.stack((m_funsim, m_seqsim, m_gossim), 0)
        d_sim = torch.stack((d_funsim, d_semsim, d_gossim), 0)
        m_attsim = self.attsim_m(m_sim)
        d_attsim = self.attsim_d(d_sim)

        # Set the diagonal to 0.0 for subsequent sampling.对角线置零，避免图构建过程中节点与自身连接
        m_final_sim = m_attsim.fill_diagonal_(fill_value=0)
        d_final_sim = d_attsim.fill_diagonal_(fill_value=0)

        return m_final_sim, d_final_sim


class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(FusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):  # in_size=out_dim * num_heads =512 8*64
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):  # [x, 2, 512] ，X是节点个数
        w = self.project(z).mean(0)  # 权重矩阵，获得元路径的重要性 [2, 1]
        # beta = torch.softmax(w, dim=0)
        beta = torch.sigmoid(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # [x,2,1]
        # delta = beta * z
        return (beta * z).sum(1)  # [x, 512]


class LocalNITCLayer(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim):
        super(LocalNITCLayer, self).__init__()
        self.local_weight_m1 = nn.Linear(i_dim, h_dim)
        self.local_weight_m2 = nn.Linear(h_dim, h_dim)
        self.local_weight_m3 = nn.Linear(h_dim, o_dim)

        self.local_weight_d1 = nn.Linear(i_dim, h_dim)
        self.local_weight_d2 = nn.Linear(h_dim, h_dim)
        self.local_weight_d3 = nn.Linear(h_dim, o_dim)

    def forward(self, inputs_row, inputs_col):
        m_embedding1 = th.relu(self.local_weight_m1(inputs_row))
        m_embedding2 = th.relu(self.local_weight_m2(m_embedding1))
        m_embedding = th.relu(self.local_weight_m3(m_embedding2))

        d_embedding1 = th.relu(self.local_weight_d1(inputs_col))
        d_embedding2 = th.relu(self.local_weight_d2(d_embedding1))
        d_embedding = th.relu(self.local_weight_d3(d_embedding2))

        return m_embedding, d_embedding


class NMRDecoder(nn.Module):
    def __init__(self, num_relations, i_dim, h_dim=128, o_dim=64):
        super(NMRDecoder, self).__init__()
        self.num_relations = num_relations
        self.l_layers = nn.ModuleList()
        for _ in range(num_relations):
            self.l_layers.append(LocalNITCLayer(i_dim, h_dim, o_dim))

        self.global_weight_m1 = nn.Linear(o_dim, o_dim)
        self.global_weight_m2 = nn.Linear(o_dim, o_dim)

        self.global_weight_d1 = nn.Linear(o_dim, o_dim)
        self.global_weight_d2 = nn.Linear(o_dim, o_dim)

    '''
    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relations):
            m_embedding, d_embedding = self.l_layers[k](inputs_row, inputs_col)

            m_embedding = th.relu(self.global_weight_m2(th.relu(self.global_weight_m1(m_embedding))))
            d_embedding = th.relu(self.global_weight_d2(th.relu(self.global_weight_d1(d_embedding))))

            outputs.append(m_embedding.mm(d_embedding.t()))
        outputs = th.cat(tuple(outputs))

        return outputs.reshape(self.num_relations, inputs_row.size(0), inputs_col.size(0))
    '''

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relations):
            m_embedding, d_embedding = self.l_layers[k](inputs_row, inputs_col)

            m_embedding = th.relu(self.global_weight_m2(th.relu(self.global_weight_m1(m_embedding))))
            d_embedding = th.relu(self.global_weight_d2(th.relu(self.global_weight_d1(d_embedding))))

            score_matrix = m_embedding @ d_embedding.T  # [N_m, N_d]
            outputs.append(score_matrix)

        outputs = th.stack(outputs, dim=0)  # [num_relations, N_m, N_d]
        outputs = outputs.permute(1, 2, 0)  # [N_m, N_d, num_relations]
        outputs = outputs.reshape(-1, self.num_relations)  # [N_m * N_d, num_classes]

        return outputs


class SuperedgeLearn(nn.Module):
    def __init__(self, param):
        super(SuperedgeLearn, self).__init__()

        self.hop = param.hop
        self.neigh_size = param.nei_size
        self.mNum = param.m_num
        self.dNum = param.d_num
        self.simClass = param.sim_class
        self.mdClass = param.md_class
        self.class_all = self.simClass + self.simClass + self.mdClass
        self.NodeFea = param.feture_size
        self.in_dim = param.in_dim
        self.hidden_dim = param.hidden_dim
        self.out_dim = param.out_dim
        self.hinddenSize = param.atthidden_fea
        self.edgeFea = param.edge_feature
        self.drop = param.Dropout

        self.actfun = nn.LeakyReLU(negative_slope=0.2)
        self.actfun2 = nn.Sigmoid()

        self.SimGet = SimMatrix(param)
        self.multi_view_attn_m = Multi_view_Attention(in_size=self.mNum, hidden_size=256, out_size=128,
                                                      num_heads=[8, 2], dropout=0.5)
        self.multi_view_attn_d = Multi_view_Attention(in_size=self.dNum, hidden_size=256, out_size=128,
                                                      num_heads=[8, 2], dropout=0.5)

        self.HGNN = HGNN_multi(
            in_dim=self.in_dim,
            hidden_dims=[self.hidden_dim],
            out_dim=self.out_dim
        )

        self.MLP = FusionMLP(input_dim=3 * self.out_dim, hidden_dim=self.hidden_dim, output_dim=self.out_dim)
        self.semantic_attention = SemanticAttention(in_size=self.out_dim)
        self.kan = KAN()
        self.h_fc = nn.Linear(384, 64)

        self.fcLinear = MLP(self.out_dim * 2, 1, self.drop, self.actfun)  # 二分类的
        # self.fcLinear = MLP(self.out_dim * 2, 4, self.drop, self.actfun)  # 多分类

    def generate_hypergraph_adjacency(self, H):
        """
        将超图发生矩阵H转换为归一化的超图邻接矩阵G
        G = Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}
        其中：
        - Dv: 节点度矩阵（每个节点参与多少条超边）
        - De: 超边度矩阵（每条超边连接多少个节点）
        """
        # 计算节点度和超边度
        Dv = torch.sum(H, dim=1)  # 节点度：[total_nodes]
        De = torch.sum(H, dim=0)  # 超边度：[total_hyperedges]

        # 构造Dv^{-1/2}和De^{-1}
        inv_Dv_sqrt = torch.pow(Dv, -0.5)
        inv_De = torch.pow(De, -1.0)

        # 避免除零导致的NaN/inf
        inv_Dv_sqrt[torch.isinf(inv_Dv_sqrt)] = 0
        inv_Dv_sqrt[torch.isnan(inv_Dv_sqrt)] = 0
        inv_De[torch.isinf(inv_De)] = 0
        inv_De[torch.isnan(inv_De)] = 0

        # 构造对角矩阵
        inv_Dv_sqrt_diag = torch.diag(inv_Dv_sqrt)
        inv_De_diag = torch.diag(inv_De)

        # 计算G = Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}
        G = torch.mm(torch.mm(inv_Dv_sqrt_diag, H), torch.mm(inv_De_diag, H.t()))
        G = torch.mm(G, inv_Dv_sqrt_diag)

        return G

    def forward(self, simData, m_d, md_node):
        # Get the similarity.
        m_sim, d_sim = self.SimGet(simData)

        # ========= STEP 1: 将 numpy 相似度矩阵转换为超图结构 =========
        # 1. 转为 numpy
        m_sim_np = m_sim.detach().cpu().numpy()
        d_sim_np = d_sim.detach().cpu().numpy()

        # 2. 构建超图（注意构建过程可以选择 is_binary=True 或 False）
        m_graph_nx, m_H_np = construct_hypergraph(m_sim_np, is_binary=False)
        d_graph_nx, d_H_np = construct_hypergraph(d_sim_np, is_binary=False)

        # 3. 转为 PyTorch Tensor 和 DGL 图
        m_H = torch.tensor(m_H_np, dtype=torch.float32).to(m_sim.device)
        d_H = torch.tensor(d_H_np, dtype=torch.float32).to(d_sim.device)
        m_graph = dgl.from_networkx(m_graph_nx).to(m_sim.device)
        d_graph = dgl.from_networkx(d_graph_nx).to(d_sim.device)
        m_graph = dgl.add_self_loop(m_graph)  # 添加自环，解决 0 入度问题
        d_graph = dgl.add_self_loop(d_graph)

        # ========= STEP 2: 超图注意力更新相似性表示 =========
        # 这里假设输入特征就是原始相似度向量（或 one-hot），也可以自定义嵌入向量
        m_input_feature = m_sim  # [mNum, feat]
        d_input_feature = d_sim  # [dNum, feat]

        # 传入 Multi_view_Attention 模块
        updated_m_sim = self.multi_view_attn_m(m_graph, m_input_feature, m_H)  # shape: [mNum, out_size]
        updated_d_sim = self.multi_view_attn_d(d_graph, d_input_feature, d_H)

        # ========= STEP 3: 构建miRNA-疾病多类型关联超图 =========
        # 获取miRNA-疾病多类型关联矩阵 m_d
        # m_d shape: [mNum, dNum]，其中：
        # 1: 上调关联, -1: 下调关联, 2: 失调关联, 0: 无关联

        # 构建miRNA超边：根据关联类型分别构建
        m_hyperedges = []
        m_key_nodes = []

        for i in range(self.mNum):
            # 找到miRNA i与疾病的上调关联 (1)
            up_diseases = torch.where(m_d[i, :] == 1)[0]
            # 找到miRNA i与疾病的下调关联 (-1)
            down_diseases = torch.where(m_d[i, :] == -1)[0]
            # 找到miRNA i与疾病的失调关联 (2)
            other_diseases = torch.where(m_d[i, :] == 2)[0]

            # 构建上调关联超边 Ei_up
            if len(up_diseases) > 0:
                hyperedge_up = [i] + [j + self.mNum for j in up_diseases]
                m_hyperedges.append(hyperedge_up)
                m_key_nodes.append(i)  # miRNA i作为键节点

            # 构建下调关联超边 Ei_down
            if len(down_diseases) > 0:
                hyperedge_down = [i] + [j + self.mNum for j in down_diseases]
                m_hyperedges.append(hyperedge_down)
                m_key_nodes.append(i)  # miRNA i作为键节点

            # 构建失调关联超边 Ei_other
            if len(other_diseases) > 0:
                hyperedge_other = [i] + [j + self.mNum for j in other_diseases]
                m_hyperedges.append(hyperedge_other)
                m_key_nodes.append(i)  # miRNA i作为键节点

        # 构建疾病超边：根据关联类型分别构建
        d_hyperedges = []
        d_key_nodes = []

        for j in range(self.dNum):
            # 找到与疾病j有上调关联的miRNA (1)
            up_mirnas = torch.where(m_d[:, j] == 1)[0]
            # 找到与疾病j有下调关联的miRNA (-1)
            down_mirnas = torch.where(m_d[:, j] == -1)[0]
            # 找到与疾病j有失调关联的miRNA (2)
            other_mirnas = torch.where(m_d[:, j] == 2)[0]

            # 构建上调关联超边 Ej_up
            if len(up_mirnas) > 0:
                hyperedge_up = [j + self.mNum] + up_mirnas.tolist()
                d_hyperedges.append(hyperedge_up)
                d_key_nodes.append(j + self.mNum)  # 疾病j作为键节点

            # 构建下调关联超边 Ej_down
            if len(down_mirnas) > 0:
                hyperedge_down = [j + self.mNum] + down_mirnas.tolist()
                d_hyperedges.append(hyperedge_down)
                d_key_nodes.append(j + self.mNum)  # 疾病j作为键节点

            # 构建失调关联超边 Ej_other
            if len(other_mirnas) > 0:
                hyperedge_other = [j + self.mNum] + other_mirnas.tolist()
                d_hyperedges.append(hyperedge_other)
                d_key_nodes.append(j + self.mNum)  # 疾病j作为键节点

        # 合并所有超边
        all_hyperedges = m_hyperedges + d_hyperedges

        # 构建超图发生矩阵H
        total_nodes = self.mNum + self.dNum
        total_hyperedges = len(all_hyperedges)
        H = torch.zeros((total_nodes, total_hyperedges), dtype=torch.float32, device=m_d.device)

        for edge_idx, hyperedge in enumerate(all_hyperedges):
            for node in hyperedge:
                H[node, edge_idx] = 1.0

        # 生成超图邻接矩阵G
        G = self.generate_hypergraph_adjacency(H)

        # 初始化超图节点特征：将updated_m_sim和updated_d_sim拼接
        # 注意：这里假设updated_m_sim和updated_d_sim的维度是[128]，需要调整到in_dim
        if updated_m_sim.shape[1] != self.in_dim:
            # 如果维度不匹配，使用线性层调整
            m_feature_adapter = nn.Linear(updated_m_sim.shape[1], self.in_dim).to(m_d.device)
            d_feature_adapter = nn.Linear(updated_d_sim.shape[1], self.in_dim).to(m_d.device)
            m_features = m_feature_adapter(updated_m_sim)
            d_features = d_feature_adapter(updated_d_sim)
        else:
            m_features = updated_m_sim
            d_features = updated_d_sim

        # 拼接miRNA和疾病特征
        hypergraph_node_features = torch.cat([m_features, d_features], dim=0)  # [mNum + dNum, in_dim]

        # 传入HGNN进行超图卷积操作
        hypergraph_embeddings = self.HGNN(hypergraph_node_features, G)  # [mNum + dNum, out_dim]

        # 分离miRNA和疾病的嵌入表示
        m_hyper_emb = hypergraph_embeddings[:self.mNum, :]  # [mNum, out_dim]
        d_hyper_emb = hypergraph_embeddings[self.mNum:, :]  # [dNum, out_dim]

        # 根据输入的md_node提取对应的节点嵌入
        m_node = md_node[:, 0]  # miRNA节点索引
        d_node = md_node[:, 1]  # 疾病节点索引

        # 提取对应的嵌入
        m_emb = m_hyper_emb[m_node]  # [batch_size, out_dim]
        d_emb = d_hyper_emb[d_node]  # [batch_size, out_dim]

        # 拼接miRNA和疾病嵌入
        md_emb = torch.cat([m_emb, d_emb], dim=1)  # [batch_size, 2*out_dim]

        h_mirnas = self.actfun(self.h_fc(m_emb[md_node[:, 0]]))
        h_diseases = self.actfun(self.h_fc(d_emb[md_node[:, 0]]))

        # 通过MLP预测
        # d_edge_score = self.fcLinear(md_emb)
        # pre_score = self.actfun2(md_edge_score).squeeze(dim=1) # 二分类的

        train_score = self.kan(h_mirnas, h_diseases)
        return train_score.view(-1)

        #return pre_score


class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.kanlayer1 = KANLinear(64, 32)
        self.kanlayer2 = KANLinear(32, 16)
        self.kanlayer3 = KANLinear(16, 1)

    def forward(self, mi_emb, di_emb):
        # mi_feat = mi_emb[mi_index]
        # di_feat = di_emb[di_index]
        pair_feat1 = mi_emb * di_emb
        pair_feat2 = self.kanlayer1(pair_feat1)
        pair_feat3 = self.kanlayer2(pair_feat2)
        pair_feat4 = self.kanlayer3(pair_feat3)
        # mi_1 = self.kanlayer1(mi_emb)
        # mi_2 = self.kanlayer2(mi_1)
        # mi_3 = self.kanlayer3(mi_2)
        #
        # di_1 = self.kanlayer1(di_emb)
        # di_2 = self.kanlayer2(di_1)
        # di_3 = self.kanlayer3(di_2)
        #
        # association_score = torch.matmul(mi_3, di_3.t())

        return torch.sigmoid(pair_feat4)


class ConstructSuperEdge(nn.Module):
    def __init__(self, edgeFea, class_all, nodeFea, hsize):
        super(ConstructSuperEdge, self).__init__()
        self.class_all = class_all
        self.nodeFea = nodeFea
        self.edgeFea = edgeFea
        self.hidden = hsize
        self.edgeLinear = nn.Linear(self.nodeFea * 2, self.edgeFea)
        self.act = nn.ReLU()
        self.Att = Attention(self.edgeFea, self.nodeFea, self.hidden)
        # 门控融合相关层
        self.edge_gate = nn.Linear(self.hidden, self.edgeFea)
        self.leakrelu = nn.LeakyReLU()

    def forward(self, mnode_list, dnode_list, mrel_list, drel_list):
        mi_emb = mnode_list[0]
        dj_emb = dnode_list[0]
        pre_md_emb = torch.cat((mi_emb, dj_emb), 2)

        edge_emb = self.edgeLinear(pre_md_emb)
        edge_emb = self.act(edge_emb)

        edge_nei_node = torch.cat((mnode_list[1], dnode_list[1]), 1)
        edge_nei_rel = torch.cat((mrel_list[0], drel_list[0]), 1)
        edge_nei = torch.cat((edge_nei_rel, edge_nei_node), 2)

        edge_nei_info = self.Att(edge_emb, edge_nei)  # [batch, hidden]
        # 门控融合
        att = torch.tanh(self.edge_gate(edge_nei_info))  # [batch, edgeFea]
        gated = edge_emb.squeeze(dim=1) * att + edge_emb.squeeze(dim=1)  # [batch, edgeFea]
        out = self.leakrelu(gated)
        return out


# ---------- HGNN 基础卷积层 ----------
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(G, x)
        return x + self.bias


# ---------- 多层 HGNN 模型 ----------
class HGNN_multi(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(HGNN_multi, self).__init__()
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(HGNN_conv(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, G):
        for i, layer in enumerate(self.layers):
            x = layer(x, G)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x  # 最后一层不加激活（保留信息）
