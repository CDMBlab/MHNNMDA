import torch
from torch import nn
import numpy as np


def Sample_nei(node_pre, rel_adj, neigh_size):
    # Determine whether the sampling size is reasonable.
    count_edge_num = torch.count_nonzero(node_pre, dim=1)
    res_min = torch.min(count_edge_num)

    nei_sample = torch.multinomial(node_pre, neigh_size, replacement=False if neigh_size <= res_min else True)
    node_sample = nei_sample
    rel_sample = rel_adj.gather(1, node_sample)
    rel_sample = (rel_sample - 1).long()

    return node_sample, rel_sample


class GetSubgraph(nn.Module):
    def __init__(self, nei_size, hop):
        super(GetSubgraph, self).__init__()
        self.neigh_size = nei_size
        self.hop = hop

    def forward(self, m_node, d_node, node_adj, rel_adj):
        # We remove node pairs during sampling.
        md_node = torch.cat((m_node, d_node), 0)
        dm_node = torch.cat((d_node, m_node), 0)
        node_edge_index = torch.nonzero(node_adj).t()

        device = node_edge_index[0].device
        node_get_pre = torch.zeros_like(node_adj)
        node_get_pre.index_put_(indices=[node_edge_index[0], node_edge_index[1]],
                                values=torch.tensor(1., device=device))
        node_get_pre.index_put_(indices=[md_node, dm_node], values=torch.tensor(0., device=device)).cuda()

        mnei_emb_list = [m_node.unsqueeze(dim=1)]
        dnei_emb_list = [d_node.unsqueeze(dim=1)]
        mrel_emb_list, drel_emb_list = [], [] # 这两个列表将保存每一跳采样到的邻居节点，供后续嵌入层使用。

        # Start randomly sampling.
        for i in range(self.hop):
            now_nei_size = self.neigh_size[i]
            nei_node_adj, rel_node_adj = Sample_nei(node_get_pre, rel_adj, now_nei_size)

            m_subnode, m_subrel = getNeiRel(mnei_emb_list[-1], nei_node_adj, rel_node_adj)
            mnei_emb_list.append(m_subnode) # 记录新采样的邻居节点
            mrel_emb_list.append(m_subrel) # 记录边的关系类型（例如边是同类、异类、某种权重）

            d_subnode, d_subrel = getNeiRel(dnei_emb_list[-1], nei_node_adj, rel_node_adj)
            dnei_emb_list.append(d_subnode)
            drel_emb_list.append(d_subrel)

        return mnei_emb_list, mrel_emb_list, dnei_emb_list, drel_emb_list


def getNeiRel(nodes_index, nei_node, nei_rel):
    # Select corresponding neighbors and relationships.
    node_index = torch.reshape(nodes_index, (-1, 1)).squeeze(dim=1)

    node_neigh = torch.index_select(nei_node, 0, node_index)
    rel_neigh = torch.index_select(nei_rel, 0, node_index)
    node_subnei = torch.reshape(node_neigh, (nodes_index.size(0), -1))
    rel_subnei = torch.reshape(rel_neigh, (nodes_index.size(0), -1))

    return node_subnei, rel_subnei
