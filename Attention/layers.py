import torch
from torch import nn
import torch.nn.functional as F

class multiHeadAttention(nn.Module):
    """
    Some explanation here
    """

    def __init__(self, n_heads, seq_len, d_V, d_K, d_Q):
        super().__init__()
        self.v_heads = nn.Linear(seq_len, n_heads*d_V, bias=False)
        self.k_heads = nn.Linear(seq_len, n_heads*d_K, bias=False)
        self.q_heads = nn.Linear(seq_len, n_heads*d_Q, bias=False)
        self.SDPA = scaledDotProductAttention(d_K)
        self.linear_out = nn.Linear(n_heads*d_Q, seq_len, bias=False)


    def forward(self, V, K, Q):
        # V, K and Q should all have shape (batch, X)

        #Apply initial linear transform to all heads
        V = self.v_heads(V)
        Q = self.v_heads(Q)
        K = self.v_heads(K)

        #Reshape for SDPA layers (need to switch heads and batch dims)
        v_s = V.shape
        V = V.view([v_s[1],v_s[0],v_s[2],v_s[3]])

        q_s = Q.shape
        Q = Q.view([q_s[1],q_s[0],q_s[2],q_s[3]])

        k_s = K.shape
        K = K.view([k_s[1],k_s[0],k_s[2],k_s[3]])

        output = self.SDPA(V,K,Q)

        #Reshape for linear layer (need to switch heads and batch dims)

        o_s = output.shape
        output = output.view([o_s[1],o_s[0],o_s[2],o_s[3]])
        output = self.linear_out(output)

        return output

class addAndNorm(nn.Module):
    """
    Some explanation here
    """

    def __init__(self, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, res, MHA_output):
        return self.norm(MHA_output + res)

class positionalEncoder(nn.Module):
    """
    Some explanation here
    """

    def __init__(self, hidden_dim_size, pos):
        super().__init__()
        self.dim = hidden_dim_size
        self.pos = pos
        self.pos_vals = self.get_positional_encoding()

    def get_positional_encoding(self):
        pos_vals = [self.pos/(10000**(2*x/self.dim_)) for x in range(self.dim_)]
        return torch.FloatTensor(pos_vals)

    def get_positional_encodings(self):
        pos_vals = self.get_positional_encoding()
        pos_vals[:,::2] = torch.sin(pos_vals[:,::2])
        pos_vals[:,1::2] = torch.cos(pos_vals[:,1::2])
        return pos_vals

    def forward(self, x):
        x = self.pos_vals+x
        return x

class positionWiseFeedForward(nn.Module):
    """
    Some explanation here
    """

    def __init__(self, seq_len, hidden_seq_len):
        super().__init__()
        self.linear1 = nn.Linear(seq_len,hidden_seq_len)
        self.linear2 = nn.Linear(hidden_seq_len,seq_len)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x):
        res = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.norm(x + res)
        return x


class scaledDotProductAttention(nn.Module):
    """
    Some explanation here
    d_K shold be the length of the Q,V,and K vectors excluding no_heads and batch size
    """

    def __init__(self, d_K):
        super().__init__()
        self.d_K = d_K

    def forward(self, V, K, Q):
        inner_attention = torch.matmul(Q, torch.transpose(K, 2,3))
        inner_attention = torch.multiply(inner_attention, 1/torch.sqrt(self.d_K))
        inner_attention = torch.matmul(F.softmax(inner_attention),V)
        return inner_attention