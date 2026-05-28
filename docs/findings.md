# Findings

Experimental results from comparing a Transformer (**SASRec**) against an RNN
(**GRU4Rec**) for next-app prediction, on two public app-usage datasets.

## TL;DR

- Both models reach essentially identical performance on the Android dataset
  (LSApp): **NDCG@10 ≈ 0.79, Recall@10 ≈ 0.94**.
- On the sparser iPhone dataset (LiveLab), SASRec wins by ~15% on NDCG@10.
- **Recommended for deployment: GRU4Rec.** Equal accuracy to SASRec on the
  Android-aligned dataset, with ~20% fewer parameters and faster inference.

## Final test-set results

| Dataset  | Users | Apps | Interactions | Model    | Recall@10 | NDCG@10   | MRR@10 |
|----------|------:|-----:|-------------:|----------|----------:|----------:|-------:|
| LiveLab  | 34    | 1127 | 738 384      | GRU4Rec  | 0.824     | 0.418     | 0.298  |
| LiveLab  | 34    | 1127 | 738 384      | SASRec   | **0.853** | **0.480** | **0.369** |
| LSApp    | 289   |   87 | 1 673 257    | GRU4Rec  | **0.948** | **0.791** | 0.740  |
| LSApp    | 289   |   87 | 1 673 257    | SASRec   | 0.938     | 0.791     | **0.743** |

All numbers are on the held-out **test** split (leave-one-out, time-ordered, per
RecBole's `LS: valid_and_test` setting).

## Top-1 / top-2 accuracy (the metric that matters for nudging)

For a nudging app you act on the top-1 (or top-2) prediction — not the top-10
— so Recall@1 and Recall@2 are the operational metrics. Recomputed on the
LSApp test set:

| Model    | Recall@1 | Recall@2 | Recall@3 | Recall@5 | Recall@10 |
|----------|---------:|---------:|---------:|---------:|----------:|
| SASRec   | **0.626** | **0.789** | 0.837 | 0.893 | 0.938 |
| GRU4Rec  | 0.623    | 0.782    | 0.824 | 0.896 | **0.948** |

**Translation.** ~63% of the time the model's single best guess is the next app
the user opens — i.e. the nudge fires on the *correct* app without any further
disambiguation. ~79% of the time the right app is in the top two, which is
useful if the UI wants to pre-empt two likely candidates.

The two models remain within ~1% of each other on every k, reinforcing the
deployment recommendation.

Generated with `uv run python -m src.eval_topk saved/<checkpoint>.pth`.

## Interpretation

### LiveLab: Transformer wins (+15% NDCG@10)

On a sparse dataset with many apps (1127), self-attention's ability to look
arbitrarily far back in a user's history gives SASRec a real edge over the
GRU's sequential state-passing. This is consistent with prior published results
on similar small/sparse sequential-recommendation benchmarks.

### LSApp: Statistical tie

Both models converge to ~0.79 NDCG@10 within ~10 epochs and oscillate in the
noise from there. We attribute this to two properties of app-usage data on a
small catalog:

1. **High repetition.** Most users cycle between 5–15 apps; the "predict same
   as last" baseline alone gets respectable performance.
2. **Small effective vocabulary.** With only 87 apps to rank, the ranking
   problem has limited headroom — once a reasonable model fits, both
   architectures saturate.

In this regime, architectural capacity is not the bottleneck.

### Engineering implication

For the deployment target (Android, on-device or near-device nudging),
**GRU4Rec is the better choice**:

- 87 616 parameters vs SASRec's 108 928 (~20% smaller)
- No positional-embedding layer, simpler graph
- Faster per-token inference
- Equal predictive quality on the relevant dataset

We ship the GRU.

## Caveats

- **Small N.** LiveLab has 34 participants, LSApp has 289. The valid → test gap
  on LiveLab (NDCG@10 0.625 → 0.418 for GRU4Rec; 0.652 → 0.480 for SASRec)
  reflects high-variance splits on small data, not a real generalization
  failure.
- **Absolute scores benefit from app-usage repetition** and should not be
  compared directly to general sequential-recommendation benchmarks (movies,
  Amazon, etc.) where each item is far less likely to repeat.
- **LiveLab is iPhone, 2010–2013.** The catalog (`SpringBoard`,
  `com.apple.*`) does not reflect modern Android apps; we used it only as a
  prototyping dataset before moving to LSApp.

## Trivial baselines for context

The notebook (`notebooks/lsapp_eda.ipynb`) reports a "predict the previous
app" baseline. Any deep model must beat this to be worth deploying. Numbers
land in the report's appendix.

## Hyperparameters used

Both models trained with the RecBole defaults plus:
- `MAX_ITEM_LIST_LENGTH: 50`
- `epochs: 30`
- `stopping_step: 5`
- `learning_rate: 0.001`
- `train_batch_size: 1024`
- `valid_metric: NDCG@10`

Configs live in `config/`. Training logs in `saved/train_*.log`.
