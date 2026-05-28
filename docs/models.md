# Model architectures

Both models solve the same problem — given a user's recent app history,
predict the next app — but go about it differently.

## Shared scaffolding

1. Each app gets a learned **embedding** (64-dim vector).
2. A user's recent history becomes a sequence of those embeddings (max 50).
3. The model produces a single "what comes next" vector.
4. That vector is scored against every app embedding; the top-k are the
   predictions.

The two models differ only in step 3.

## GRU4Rec — the RNN baseline

Reference: Hidasi et al., *Session-Based Recommendations with Recurrent Neural
Networks*, 2016.

Reads the sequence **one app at a time, left to right**, maintaining a hidden
state that is a running summary of the history seen so far.

```
[Facebook] → GRU → [WhatsApp] → GRU → [YouTube] → GRU → predict
                ↑               ↑                ↑
            hidden_t1       hidden_t2        hidden_t3
```

Each GRU cell takes `(previous hidden, current embedding)` and produces a new
hidden state using two gates:

- **Update gate** — how much of the previous hidden state to retain.
- **Reset gate** — how much of the previous hidden state to forget when
  blending in the new input.

The hidden state after the final input is the prediction vector.

- **Trainable params (LSApp config):** 87 616
- **Strengths:** simple, parameter-efficient, captures recency naturally.
- **Weaknesses:** information from early in the sequence has to survive being
  passed through every later cell; long-range memory degrades.

## SASRec — the Transformer baseline

Reference: Kang & McAuley, *Self-Attentive Sequential Recommendation*, 2018.

Reads the **entire history at once** and uses **causal self-attention** to let
each position look directly at every earlier position.

For each position the model computes three vectors:

- **Query** ("what am I looking for?")
- **Key** ("what do I represent?")
- **Value** ("what info do I carry?")

The Query of the "next slot" is compared against all earlier Keys; the
resulting attention weights are used to take a weighted sum of the Values.
This happens in **multiple heads in parallel** so different heads can learn
different relations.

```
[Facebook, WhatsApp, YouTube, …]   →   causal self-attention   →   predict
         ↑___________↑________↑
         every slot can attend to every earlier slot
```

Because attention is order-agnostic on its own, SASRec adds a **positional
embedding** to each input ("I am position 7"). The attention block is stacked
**twice** by default.

- **Trainable params (LSApp config):** 108 928
- **Strengths:** any position can attend to any earlier position with no
  degradation; parallelizable.
- **Weaknesses:** more parameters; needs positional embeddings; can overfit
  small datasets without strong dropout.

## Quick mental analogy

- **GRU4Rec:** like someone summarizing a book chapter-by-chapter, holding a
  running mental summary.
- **SASRec:** like someone with the whole book open, free to flip back to any
  earlier page at any time.

## Why they tie on LSApp

App-usage sequences are highly repetitive (most users cycle between 5–15
apps), and LSApp has only 87 unique apps in total. Both architectures hit the
same performance ceiling around NDCG@10 ≈ 0.79. See
[`findings.md`](findings.md) for the full discussion.
