import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        
        # this actually not embedding but look up table simplified for this purpose
        # of training next token pred without prev context 
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # map word idx to vector of vocab_size
        logits = self.embeddings(idx) # batch, block_size (T - time / seq len), vocab_size (C - channel)
        
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
            
            logits, loss = self.forward(idx)

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
    
    batch_size = 32
    block_size = 64
    lr = 1e-3
    max_iters = 10000
    eval_iters = 10
    eval_interval = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from data.shakespeare.prepare import ShakespeareDataset
    dataset = ShakespeareDataset()
    x, y = dataset.get_batch(batch_size, block_size, 'train')
    
    model = BigramLanguageModel(dataset.vocab_size)
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