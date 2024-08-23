from torch import nn


class AbstractionTransformer(nn.Module):
    def __init__(self, d_model, output_vocab, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(AbstractionTransformer, self).__init__()
        self.d_model = d_model
        self.output_vocab = output_vocab
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.output_embedding = nn.Embedding(output_vocab, d_model)
        self.transformer_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
    
    