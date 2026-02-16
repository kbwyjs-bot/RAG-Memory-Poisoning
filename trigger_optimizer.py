import torch
from torch import nn
import tiktoken
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def load_pom_dataset(path: str, max_samples: int | None = None):
    """
    Loads attacker prompts from pom.txt exactly as they are.
    Each non-empty line is treated as one sample.
    """
    samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                samples.append(line)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples
def append_trigger_text(text_data, trigger_text: str):
    """
    Appends the same trigger string to each sentence.
    
    text_data: list[str]
    trigger_text: str
    
    returns: list[str]
    """
    poisoned = []

    for sentence in text_data:
        # Ensure clean spacing between sentence and trigger
        combined = sentence.rstrip() + " " + trigger_text.lstrip()
        poisoned.append(combined)

    return poisoned

data = load_pom_dataset(
    r"C:\Users\l0rdTheMerciful\Desktop\res LLM\RAG\RAG poisoning\pom.txt"
)

trigger = 'lordTheMerciful'
def load_embeding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')
    model.eval()
    return model
def embedding_shift(clean, poisoned):
    return poisoned - clean

#0. loadin model
model = load_embeding_model()
#1. embedding clean token ids
with torch.no_grad():
    embeddings = model.encode(data, convert_to_tensor = True)
#2. embedding poisoned data
poison = append_trigger_text(data, trigger)
with torch.no_grad():
    poisoned_embeddings = model.encode(poison, convert_to_tensor = True)
shift = embedding_shift(embeddings, poisoned_embeddings)

#OUTPUT: First 10 shift Norms:[0.6095402836799622, 0.25920748710632324, 0.27148568630218506]
# average shift Norm:0.3800778388977051\

shift_norms = torch.norm(shift, dim = 1)
print(f"First 10 shift Norms:{shift_norms[:10].tolist()}")
print(f"average shift Norm:{shift_norms.mean().item()}")

#checking cosine similarity
ref = shift[0]
others = shift[1:6]
print(f'reference shift:{ref.shape}')
print(f"others:{others.shape}")
cos_values = F.cosine_similarity(others, ref.unsqueeze(0), dim=1)
print(f'Cosine Similarity vs Shift[0]:{cos_values.tolist()}')