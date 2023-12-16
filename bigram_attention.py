import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)


class HeadAttention(nn.Module):
    
    def __init__(self, block_size, embedding_dim, head_size):
        super().__init__()
        self. key = nn. Linear (embedding_dim, head_size, bias=False)
        self. query = nn. Linear (embedding_dim, head_size, bias=False)
        self. value = nn. Linear (embedding_dim, head_size, bias=False)
        self. register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        
        B, T, C = x.shape
        
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
                
        # compute attention scores ("affinities")
        wei = q @ k. transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei. masked_fill(self. tril[:T, : T] == 0, float ('-inf')) # (B, T, T)
        wei = F. softmax (wei, dim=-1) # (B, T, T
        
        # perform the weighted aggregation of the values
        v = self. value(x) # (B,T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        
        return out
    
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, block_size, embedding_dim, head_size, num_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(block_size, embedding_dim, head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        # x: (B, T, C)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, C * head_size)
        return out


class Block(nn.Module):
    
    def __init__(self, block_size, embedding_dim, head_size, num_heads):
        super().__init__()
        self.self_attention_heads = MultiHeadAttention(block_size, embedding_dim, head_size, num_heads)
        self.ffwd = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # x: (B, T, C)
        x = self.self_attention_heads(x)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(BigramLanguageModel, self).__init__()
        
        # this actually not embedding but look up table simplified for this purpose
        # of training next token pred without prev context 
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.self_attention_heads = MultiHeadAttention(block_size, embedding_dim, head_size=embedding_dim // 4, num_heads=4)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape # batch_size, block_size
        
        # map word idx to vector of vocab_size
        token_embeddins = self.embeddings(idx) # batch, block_size (T - time / seq len), vocab_size (C - channel)
        position_embeddings = self.position_embedding(torch.arange(T, device=token_embeddins.device))
        x = token_embeddins + position_embeddings # B, T, C
        
        x = self.self_attention_heads(x) # B, T, C
        
        logits = self.lm_head(token_embeddins) # B, T, C
        
        if targets is None:
            # this inference with no need of loss (training)
            loss = None
        else:
            # we need to reshape logits and targets to be 2D so that we can use cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
        
            # how cross entropy loss works:
            # 1. apply softmax to logits 
            # softmax - turns logits (vector of vocabsize) into probability distribution, sum of all probs = 1
            # 2. claculate -ln(taget's softmax value) - -ln(0) = a lot, -ln(1) = 0
            loss = F.cross_entropy(logits, targets) # with no training random pick of char, -ln(1/vocab_size(65)) = 4,7
        
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            # cut the last block_size tokens from the sequence
            idx_cond = idx[:, -block_size:] # B, T
            
            logits, loss = self.forward(idx_cond)

            # get the last token's logits
            logits = logits[:, -1, :] # become B, C
            
            # get probability distribution for the last token of the given batch
            probs = F.softmax(logits, dim=-1)  #
            
            # samples based on the probability distribution, [0.2, 0.9] = first has prob 0.2, second 0.8
            new_token = torch.multinomial(probs, num_samples=1)
            
            # append sampled token to the sequence of characters
            idx = torch.cat((idx, new_token), dim=-1)
            
        return idx
    
    @torch.no_grad() # we dont intend to do backprop here
    def estimate_loss(self, eval_iters: int):
        out = {}
        model.eval() # switching model to eval mode
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = dataset.get_batch(batch_size, block_size, split)
                logits, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train() # switching model back to train mode
        return out


if __name__ == "__main__":
    
    batch_size = 64
    block_size = 32
    lr = 1e-3
    max_iters = 10000
    eval_iters = 10
    eval_interval = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding_dim = 128
    
    from data.shakespeare.prepare import ShakespeareDataset
    dataset = ShakespeareDataset()
    x, y = dataset.get_batch(batch_size, block_size, 'train')
    
    model = BigramLanguageModel(dataset.vocab_size, embedding_dim)
    model = model.to(device)
    
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = model.generate(idx, 100)
    print('Before training: ')
    print(dataset.decode(pred.squeeze().tolist()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for iter in range(max_iters):
        
        x, y = dataset.get_batch(batch_size, block_size, 'train')
        x, y = x.to(device), y.to(device)
        
        # inference
        logits, loss = model(x, y)
        
        # Zero out the gradients from the previous batch
        optimizer.zero_grad(set_to_none=True)
        
        # Compute gradients for this batch
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        if iter % eval_interval == 0:
            losses = model.estimate_loss(eval_iters)
            print(f'iter {iter} | train loss {losses["train"]:.5f} | val loss {losses["val"]:.5f}')
    
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = model.generate(idx, 100)
    print('After training: ')
    print(dataset.decode(pred.squeeze().tolist()))