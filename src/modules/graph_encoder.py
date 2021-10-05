import dynet as dy

from src.dynet_modules import Attention
from src.utils import Encoder


class GraphEncoder(Encoder):
    def __init__(self, args, model, key='graph', n_layers=4):
        super().__init__(args, model)
        self.key = key
        self.special = self.model.add_lookup_parameters((1, self.args.token_dim))

        self.graph_positions = self.model.add_lookup_parameters((4, self.args.token_dim))

        self.n_layers = n_layers
        self.attentions = [
            Attention(self.model, self.args.token_dim, self.args.token_dim)
            for i in range(n_layers)
        ]

        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

    def encode(self, sent, pred=False):
        sent.root.vecs['feat'] = self.special[0]

        n_tokens = len(sent.tokens)
        # making a graph "positional" table
        # 0 --- itself, 1 --- parent, 2 --- child, 3 --- other
        rel_positions = [[3 for i in range(n_tokens)] for j in range(n_tokens)]

        ancs = dict()

        def dfs(u):
            deps = u['pdeps'] if pred else u['deps']
            res = [v for v in deps]
            for v in deps:
                res.extend(dfs(v))
            ancs[u] = res
            return res

        dfs(sent.root)

        for u in ancs:
            u_i = u['tid']
            rel_positions[u_i][u_i] = 0
            for v in ancs[u]:
                v_i = v['tid']
                rel_positions[u_i][v_i] = 1
                rel_positions[v_i][u_i] = 2

        feats = [None for _ in range(n_tokens)]
        for u in sent.tokens:
            u_i = u['tid']
            u_feat = u.vecs['feat']
            feats[u_i] = u_feat

        for u in sent.tokens:
            u_i = u['tid']
            attn_feats = []
            for v in sent.tokens:
                v_i = v['tid']
                rel_emb = self.graph_positions[rel_positions[u_i][v_i]]
                attn_feats.append(rel_emb + feats[v_i])
            attn_feats = dy.concatenate_cols(attn_feats)
            vec = feats[u_i]
            for i in range(self.n_layers):
                vec = vec + self.attentions[i].encode(attn_feats, vec)  # TODO: add LayerNorm
            u.vecs[self.key] = vec
