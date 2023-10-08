import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.activations import ACT2FN



class TensorcontextSeq2Mat(nn.Module):      
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = config.num_attention_heads
        self.d = config.num_d
        self.W = nn.Linear(3 * config.hidden_size+self.d, config.hidden_size)     
        self.V = nn.Parameter(torch.Tensor(self.d, config.hidden_size, config.hidden_size))     
        self.norm   = T5LayerNorm(config.hidden_size, config.layer_norm_eps)    
        self.activation = ACT2FN[config.hidden_act]
        self.init_weights()

    def init_weights(self):    
        if self.config.model_type == 'bart' or self.config.model_type == 't5':  
            self.V.data.normal_(mean=0.0, std=0.02)
        else:
            self.V.data.normal_(mean=0.0, std=self.config.initializer_range)

    def rntn(self, x, y, xmat):     
        max_len = xmat.shape[1]     
        xmat_t = xmat.transpose(1, 2)       
        batch_size = xmat.shape[0]      
        context = torch.ones_like(x).to('cuda')

        for i in range(max_len):                                
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)    
            xmat_t = torch.max(xmat_t[:, :, :max_len-i], diag)
            bb = [[b] for b in range(batch_size)]   

            linexup = [[j for j in range(max_len-i)] for b in range(batch_size)]    
            lineyup = [[j+i for j in range(max_len-i)] for b in range(batch_size)] 
            linexdown = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len-i)] for b in range(batch_size)]   
            context[bb, linexup, lineyup, :] = xmat_t.permute(0, 2, 1)
            context[bb, linexdown, lineydown, :] = xmat_t.permute(0, 2, 1)

        t = torch.cat([x, y, context], dim=-1)
        xvy = torch.einsum('b m n p, k p d, b m n d -> b m n k', x, self.V, y)      
        t = torch.cat([t, xvy], dim=-1)     
        tw = self.W(t)      
        return tw       

    def forward(self, x, y):     

        xmat = x    

        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])

        t = self.rntn(x, y, xmat)    
        t = self.activation(t)

        return t    

