import torch
import torch.nn as nn
import torch.nn.functional as F
vocab = {'good':0,'bad':1, 'movie':2, "not":3, "very":4}
V = len(vocab)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(V, 6)
        self.fc = nn.Linear(6, 2)
    
    def forward(self, x_ids):
        emb = self.embedding(x_ids)
        pooled = emb.mean(dim = 0)
        logits = self.fc(pooled)
        return logits, emb
    
model = TinyModel()

target = torch.tensor([1])
sentence = ['good', 'movie']
x = [vocab[w] for w in   sentence]
x = torch.tensor(x)
logits, emb = model(x)
probs = F.softmax(logits, dim = 0)
print(f"logits are:{logits}")
print(f"probabilities: {probs}")
loss = F.cross_entropy(logits.unsqueeze(0), target)
print('\n------------------ Before HOTFLIP-------------')
print(f'logits: {logits.detach().tolist()}')
print(f'Probabilities{probs.detach().tolist()}')
print(f"Loss:{loss.detach()}")
id_to_word = {i:w for w, i in vocab.items()}
model.zero_grad()
# this line tells pytorch like hey i want to keep the gradients for this non leaf tensor
emb.retain_grad()
loss.backward()

#capturing the gradients for all words 
grad_per_position = emb.grad.detach()

print('\n ------gradiet per position (each row = one word in sentence)---------')
for i in range(len(x)):
    print(f'pos{i} word: {id_to_word[x[i].item()]} grad = {grad_per_position[i].tolist()}')