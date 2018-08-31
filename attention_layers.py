from __future__ import unicode_literals, print_function, division
import torch
from torch import nn

from SubLayers import MultiHeadAttention,PositionwiseFeedForward,BottleSoftmax
import numpy as np

def extract_last(input,input_lengths):

	idx = (torch.LongTensor(input_lengths) - 1).view(-1, 1).expand(len(input_lengths),input.size()[-1]).unsqueeze(1)
	if config.use_gpu:
		idx = idx.cuda()
	return torch.gather(input,dim=1,index=idx).squeeze(1)

def position_encoding_init(n_position, d_pos_vec):
	''' Init the sinusoid position encoding table '''

	# keep dim 0 for padding token position encoding zero vector
	position_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
		if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

	position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
	position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
	return torch.from_numpy(position_enc).type(torch.FloatTensor)

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Google_self_attention(nn.Module):
	def __init__(self,dropout=0.5,add_pos=True):
		super(Google_self_attention,self).__init__()
		self.add_pos = add_pos
		
		self.position_enc = nn.Embedding(config.max_enc_steps+1,config.emb_dim, padding_idx=config.PAD_TOKEN)
		self.position_enc.weight.data = position_encoding_init(config.max_enc_steps+1, config.emb_dim)
		
		d_k,d_v = config.d_k,config.d_v
		d_model,d_inner_hid,n_head = config.d_model,config.d_inner_hid,config.n_head
		n_layers = config.n_layers

		self.layer_stack = nn.ModuleList([
				EncoderLayer(d_model,d_inner_hid,n_head, d_k, d_v, dropout=dropout)
				for _ in range(n_layers)])

	def forward(self,seq_emb,seq_pos,seq_mask,return_attns=False):
		
		if seq_pos is not None:
			pos_emb = self.position_enc(seq_pos)
			seq_emb = seq_emb + pos_emb
		
		if return_attns:
			enc_slf_attns = []

		enc_output = seq_emb
		enc_slf_attn_mask = seq_mask
		
		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(
				enc_output, slf_attn_mask=enc_slf_attn_mask)
			if return_attns:
				enc_slf_attns += [enc_slf_attn]

		if return_attns:
			return enc_output, enc_slf_attns
		else:
			return enc_output,None
def isnan(x):
    return torch.isnan(x).sum()

class Cross_attention(nn.Module):
	def __init__(self):
		super(Cross_attention,self).__init__()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,D,doc_mask,doc_lens,Q,query_mask):
		
		A_d = torch.bmm(D,Q.transpose(1,2))
		A_q = torch.bmm(Q,D.transpose(1,2))

		A_d = self.softmax(A_d.data.masked_fill_(query_mask, -float('inf')))
		A_q = self.softmax(A_q.data.masked_fill_(doc_mask,-float('inf')))

		C_q = torch.bmm(A_q,D)
		C_d = torch.bmm(A_d,Q)
		
		query_output = torch.cat([Q,C_q],dim=-1)
		C_D = torch.bmm(A_d,query_output)
		doc_output = torch.cat([C_D,D],dim=-1)
		return doc_output,query_output,extract_last(doc_output,doc_lens)

if __name__ == '__main__':

	a = np.zeros((64,400))
	for i in range(len(a)):
		a[i] = np.arange(1,401)
	print(a.shape)
	a = torch.from_numpy(a).long()
	emb = nn.Embedding(401,64,padding_idx=0)
	vec = emb(a)
	print(vec.size())