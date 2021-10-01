import dynet as dy

from src.dynet_modules import Attention
from src.utils import Encoder


class GraphEncoder(Encoder):
    def __init__(self, args, model, key='graph'):
        super().__init__(args, model)
        self.key = key
        self.special = self.model.add_lookup_parameters((1, self.args.token_dim))

        self.graph_positions = self.model.add_lookup_parameters((4, self.args.token_dim))
        self.attention = Attention(self.model, self.args.token_dim, self.args.token_dim)

        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

    def encode(self, sent, pred=False):
        sent.root.vecs['feat'] = self.special[0]

        feats = [None for _ in range(len(sent.tokens))]
        for u in sent.tokens:
            u_i = u['tid']
            u_feat = u.vecs['feat']
            feats[u_i] = u_feat

        for u in sent.tokens:
            u_i = u['tid']
            attn_feats = []
            for v in sent.tokens:
                v_i = v['tid']
                rel_emb = self.graph_positions[sent['rel_positions'][u_i][v_i]]
                attn_feats.append(rel_emb + feats[v_i])
            u.vecs[self.key] = self.attention.encode(dy.concatenate_cols(attn_feats), feats[u_i])
