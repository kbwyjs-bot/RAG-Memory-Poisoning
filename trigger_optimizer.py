import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def load_pom_dataset(path: str, max_samples: int | None = None):
    samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                samples.append(line)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, encoder, max_length=128):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        sent_emb = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        sent_emb = F.normalize(sent_emb, p=2, dim=1)
    return sent_emb


def encode_from_embeds(inputs_embeds, attention_mask, encoder):
    outputs = encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    sent_emb = mean_pool(outputs.last_hidden_state, attention_mask)
    sent_emb = F.normalize(sent_emb, p=2, dim=1)
    return sent_emb


def build_poisoned_embeds_batch(texts, trigger_ids, tokenizer, encoder, max_length=128):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length - len(trigger_ids),
        return_tensors="pt"
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    embedding_layer = encoder.get_input_embeddings()

    text_embeds = embedding_layer(input_ids)  # [B, T, D]

    trigger_embeds = embedding_layer(trigger_ids)  # [L, D]
    trigger_embeds = trigger_embeds.unsqueeze(0).repeat(input_ids.size(0), 1, 1)
    trigger_embeds.retain_grad()

    trigger_mask = torch.ones((input_ids.size(0), len(trigger_ids)), dtype=attention_mask.dtype)

    poisoned_embeds = torch.cat([text_embeds, trigger_embeds], dim=1)
    poisoned_attention_mask = torch.cat([attention_mask, trigger_mask], dim=1)

    return poisoned_embeds, poisoned_attention_mask, trigger_embeds


def compute_trigger_gradients(texts, clean_embeddings, trigger_ids, tokenizer, encoder):
    poisoned_embeds, poisoned_attention_mask, trigger_embeds = build_poisoned_embeds_batch(
        texts, trigger_ids, tokenizer, encoder
    )

    poisoned_embeddings = encode_from_embeds(
        poisoned_embeds, poisoned_attention_mask, encoder
    )

    shift = poisoned_embeddings - clean_embeddings
    shift_norms = torch.norm(shift, dim=1)
    objective = shift_norms.mean()

    encoder.zero_grad()
    objective.backward()

    grad = trigger_embeds.grad.mean(dim=0).detach()
    return objective.item(), grad


def hotflip_update(trigger_ids, grad, embedding_matrix):
    best = {"score": -1e9, "pos": None, "to_id": None}

    for pos in range(len(trigger_ids)):
        current_id = trigger_ids[pos].item()
        e_current = embedding_matrix[current_id]
        g = grad[pos]

        for cand_id in range(embedding_matrix.size(0)):
            if cand_id == current_id:
                continue

            e_cand = embedding_matrix[cand_id]
            delta = e_cand - e_current
            score = torch.dot(g, delta).item()

            if score > best["score"]:
                best["score"] = score
                best["pos"] = pos
                best["to_id"] = cand_id

    new_trigger_ids = trigger_ids.clone()
    new_trigger_ids[best["pos"]] = best["to_id"]
    return new_trigger_ids, best


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)
encoder.eval()

data = load_pom_dataset(
    r"C:\Users\l0rdTheMerciful\Desktop\res LLM\RAG\RAG poisoning\pom.txt",
    max_samples=16
)

clean_embeddings = encode_texts(data, tokenizer, encoder)

init_trigger_tokens = ["the", "the", "the"]
trigger_ids = torch.tensor(tokenizer.convert_tokens_to_ids(init_trigger_tokens), dtype=torch.long)

E = encoder.get_input_embeddings().weight.detach()

for step in range(5):
    objective_value, grad = compute_trigger_gradients(
        data, clean_embeddings, trigger_ids, tokenizer, encoder
    )

    trigger_ids, best = hotflip_update(trigger_ids, grad, E)
    trigger_tokens = tokenizer.convert_ids_to_tokens(trigger_ids.tolist())

    print(f"\nStep {step + 1}")
    print("Objective:", objective_value)
    print("Best position:", best["pos"])
    print("New trigger tokens:", trigger_tokens)