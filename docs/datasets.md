# Datasets

We evaluated two public app-usage datasets and ship the model trained on the
second (LSApp).

## LSApp — primary

- **Source:** <https://github.com/aliannejadi/LSApp>
- **Paper:** Aliannejadi et al., *Context-Aware Target Apps Selection and
  Recommendation for Enhancing Personal Mobile Assistants*, 2021.
- **Platform:** Android
- **Period:** 2018
- **Participants:** 292 (289 after sparsity filtering)
- **Unique apps:** 87
- **Interactions used:** 1 673 257 (`Opened` events only)
- **File:** `data/raw/lsapp.tsv` (after extracting `lsapp.tsv.gz`)

Schema:
```
user_id  session_id  timestamp  app_name  event_type
```

`event_type` is `Opened` or `Closed`. We keep only `Opened` rows for
sequential next-app prediction; `Closed` rows are used by the notebook to
compute session durations.

Converter: `src/convert_lsapp.py`.

## LiveLab — prototype only, not deployed

- **Source:** <https://yecl.org/livelab/traces.html>
- **Paper:** Shepard et al., *LiveLab: measuring wireless networks and
  smartphone users in the field*, SIGMETRICS 2010.
- **Platform:** iPhone
- **Period:** 2010–2013
- **Participants:** 34
- **Unique apps:** 1127
- **Interactions used:** 738 384 (after dropping iOS `SpringBoard` and
  sparsity filtering)
- **File:** `data/raw/appusage.sql` (inside `apps.tar.gz`)

Schema (from `apps.tar.gz / appusage.sql`):
```
id  uid  name  time (POSIX seconds)  duration
```

Converter: `src/convert_livelab.py`. Parses the SQL dump directly so it works
regardless of whether the dump targets MySQL or SQLite.

We used LiveLab for early prototyping. It was retired in favour of LSApp once
we realised the platform/era mismatch with the project's Android target.

## Why not merge LiveLab + LSApp?

- App identifiers don't match (`com.facebook.Facebook` vs `Facebook`); only
  ~20–30 apps overlap, and aligning them requires hand-built mapping.
- Different eras (2012 iPhone vs 2018 Android) — different app ecosystems.
- Sequential models work on per-user histories; more users help, but the
  unmapped item embeddings would not transfer.
- Methodologically muddies the report: "we trained on Android sequential app
  usage" is clean; "we trained on a Frankenstein of iPhone-2012 and
  Android-2018" is not.

## Datasets considered but skipped

- **Tsinghua App Usage Dataset** — 1000 users, 2000 apps, Android. Excellent
  fit but behind an IEEE DataPort subscription paywall (free only to IEEE
  members). <https://fi.ee.tsinghua.edu.cn/appusage/>
- **MobileRec** — 19.3M interactions, 700k users. Looks attractive but the
  data is Google Play Store *review/install* history, not on-device foreground
  sequences. Wrong shape for next-app prediction.
  <https://arxiv.org/abs/2303.06588>

## On-device logging — feasibility

Real on-device app-usage logging is platform-restricted:

- **iOS:** no public API for third-party apps. Screen Time data is not
  accessible. Dead end.
- **Android:** requires a signed app declaring `PACKAGE_USAGE_STATS` and the
  user granting "Usage access" in Settings. Technically possible but out of
  scope for this project's timeline.

For the demo we use LSApp directly. In production, a small Android logger
would write events to the same `/log_event` endpoint.
