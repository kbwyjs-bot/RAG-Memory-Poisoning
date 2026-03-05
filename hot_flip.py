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

emb.retain_grad()
loss.backward()


grad_per_position = emb.grad.detach()

print('\n ------gradiet per position (each row = one word in sentence)---------')
for i in range(len(x)):
    print(f'pos{i} word: {id_to_word[x[i].item()]} grad = {grad_per_position[i].tolist()}')

E = model.embedding.weight.detach()  

best = {
    "score": -1e9,
    "pos": None,
    "from_id": None,
    "to_id": None,
}

for pos in range(len(x)):
    current_id = x[pos].item()
    g = grad_per_position[pos]         
    e_current = E[current_id]           

    for cand_id in range(V):
        #V is the length of vocabulary
        if cand_id == current_id:
            continue

        e_cand = E[cand_id]
        delta = e_cand - e_current
        score = torch.dot(g, delta).item()

        if score > best["score"]:
            best["score"] = score
            best["pos"] = pos
            best["from_id"] = current_id
            best["to_id"] = cand_id

print("\n--- HOTFLIP DECISION ---")
print("Best position to flip:", best["pos"])
print("Replace:", id_to_word[best["from_id"]], "->", id_to_word[best["to_id"]])
print("HotFlip score (bigger = more increases loss):", best["score"])


x_flipped = x.clone()
x_flipped[best["pos"]] = best["to_id"]

logits2, _ = model(x_flipped)
probs2 = F.softmax(logits2, dim=0)
loss2 = F.cross_entropy(logits2.unsqueeze(0), target)
def decode(x):
    words = []
    for i in x:
        word = id_to_word[i.item()]
        words.append(word)
    return words


print("\n--- AFTER HOTFLIP ---")
print("Flipped sentence:", decode(x_flipped))
print("Logits:", logits2.detach().tolist())
print("Probabilities [neg, pos]:", probs2.detach().tolist())
print("Loss:", float(loss2.detach()))