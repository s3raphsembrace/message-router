# Architecture — Message Notification Router

Seven diagrams, from the whole pipeline down to the decisions that are easy to get
wrong. Every box maps to a real module; every number is measured, not estimated.

---

## 1. System overview

The four layers, what flows between them, and where state is cached.

```mermaid
flowchart TB
    subgraph DATA["dataset/"]
        MSG["messages.csv<br/>110 rows to route"]
        CTX["users · groups · group_members<br/>business_accounts · user_business_history<br/>message_history · message_events<br/>daily_notification_summary"]
        MED["media/images (20) · media/audio (13)<br/>images.csv · voice_notes.csv"]
        SMP["sample_messages.csv<br/>30 labelled — SCORING ONLY"]
    end

    subgraph L1["LAYER 1 — code/preprocess/"]
        P1["run.py → gemini.py<br/>native audio + vision"]
        MC[("cache/media_interpretations.json<br/>keyed on media_id · 30 entries")]
    end

    subgraph L2["LAYER 2 — code/context/"]
        LOAD["loaders.Dataset<br/>index all CSVs once"]
        AGG["aggregates<br/>reaction stats · baselines"]
        SIG["signals.extract<br/>facts the model cannot verify"]
        RET["retrieve.shortlist<br/>filter → rank → top-K"]
        ASM["assemble.build_context<br/>1400-token budget"]
    end

    subgraph L3["LAYER 3 — code/router/"]
        PR["prompt.py<br/>4-stage ladder · 2185 tok"]
        CL["client.RouterClient<br/>gemini-3.5-flash-lite · temp 0"]
        RC[("cache/router_responses.json<br/>keyed on prompt hash")]
        VAL["validate.py → route.py<br/>validate · 1 re-ask · safe default"]
    end

    subgraph L4["LAYER 4 — code/guard/"]
        RUL["rules.py<br/>5-rule ladder, monotonic"]
        AUD[("cache/override_audit.csv<br/>model verdict vs rule verdict")]
    end

    OUT["dataset/output.csv<br/>110 rows"]
    VS["validate_submission.py<br/>independent contract check"]
    ZIP["code.zip"]

    MED --> P1 --> MC
    MSG --> LOAD
    CTX --> LOAD
    MC --> LOAD
    LOAD --> AGG --> SIG
    LOAD --> RET
    SIG --> ASM
    RET --> ASM
    AGG --> ASM
    ASM --> PR --> CL
    CL <--> RC
    CL --> VAL
    VAL --> RUL --> AUD
    RUL --> OUT --> VS --> ZIP

    SMP -.->|"labels stripped<br/>never reach L2/L3"| EV["code/evaluation/<br/>score · golden · ledger"]
    OUT -.-> EV

    classDef cache fill:#fff4e6,stroke:#d9822b
    classDef gate fill:#e8f5e9,stroke:#2e7d32
    class MC,RC,AUD cache
    class VS,ZIP gate
```

---

## 2. Module map

What lives where. Test suites are colocated with the layer they cover.

```mermaid
flowchart LR
    ROOT["message-router/"]

    ROOT --> CODE["code/"]
    ROOT --> DSET["dataset/"]
    ROOT --> CACHE["cache/"]

    CODE --> ENT["main.py — pipeline entry<br/>validate_submission.py<br/>package.py<br/>envload.py"]
    CODE --> PRE["preprocess/<br/>schema · media_index · cache<br/>prompts · gemini · run<br/>test_preprocess (54)"]
    CODE --> CON["context/<br/>loaders · aggregates · textutil<br/>signals · retrieve · assemble · run<br/>test_context (100)"]
    CODE --> ROU["router/<br/>prompt · client · validate<br/>decision · route · writer · run<br/>test_router (148)"]
    CODE --> GUA["guard/<br/>rules · apply · audit<br/>test_guard (57)"]
    CODE --> EVA["evaluation/<br/>metrics · ledger · golden<br/>main · run_golden · runs.csv<br/>test_eval (119)"]
    CODE --> ANA["analysis/<br/>collision_check<br/>rule_validation"]

    classDef tests fill:#eef2ff,stroke:#4f46e5
    class PRE,CON,ROU,GUA,EVA tests
```

**478 assertions across five suites**, all offline — no network, no API key.
`analysis/` holds the two scripts that justify every threshold in the system.

---

## 3. Layer 1 — media interpretation

Every outcome is a typed record. The status decides whether it is cached, because
a data problem is permanent and an environment problem is not.

```mermaid
stateDiagram-v2
    [*] --> Resolve

    Resolve --> missing_index: "id absent from<br/>images.csv / voice_notes.csv"
    Resolve --> missing_file: "indexed, not on disk"
    Resolve --> unreadable: "empty or unopenable"
    Resolve --> Interpret: "file OK"

    Interpret --> no_api_key: "no credentials"
    Interpret --> api_error: "call failed after retries"
    Interpret --> bad_response: "unexpected shape"
    Interpret --> ok: "transcript / OCR + layout"

    missing_index --> Cached
    missing_file --> Cached
    unreadable --> Cached
    ok --> Cached

    no_api_key --> NotCached
    api_error --> NotCached
    bad_response --> NotCached

    Cached --> [*]: "permanent — a property of the data"
    NotCached --> [*]: "transient — retry when the env is fixed"

    note right of NotCached
        Caching these would poison
        every future run: adding a key
        would keep serving empty
        placeholders forever.
    end note
```

Voice notes yield a verbatim transcript plus a one-line intent summary. Images
yield OCR text, a short description, and a layout class of `stock_promo` /
`personal_screenshot` / `other`. **A row whose media fails routes on metadata
alone — it never fails the run.**

---

## 4. Layer 2 — context assembly

Which table feeds which block. Nothing is dumped raw: 756 daily-load rows become
three numbers, and a user's history becomes a ranked shortlist of at most six.

```mermaid
flowchart LR
    U["users.csv"] --> UB["user<br/>quiet hours · 30d behaviour<br/>baseline_open_share"]
    DN["daily_notification_summary"] --> UB
    G["groups.csv"] --> SB["sender<br/>identity + structure"]
    GM["group_members.csv"] --> SB
    B["business_accounts.csv"] --> SG
    UBH["user_business_history.csv"] --> SG
    GM --> SG["signals<br/>scam conjunction · opt-out<br/>mute state · direct address<br/>repetition · quiet hours"]

    MH["message_history.csv"] --> RA["rapport_with_this_sender<br/>per-counterpart reactions"]
    ME["message_events.csv"] --> RA
    MH --> EC["evidence_candidates<br/>numbered top-K"]
    ME --> EC
    L1["media interpretation"] --> MSGB["message<br/>authored text kept separate<br/>from model-derived text"]

    UB --> CTX["DecisionContext<br/>min 234 · p50 632 · max 936 tokens"]
    SB --> CTX
    SG --> CTX
    RA --> CTX
    EC --> CTX
    MSGB --> CTX

    classDef derived fill:#f3e8ff,stroke:#7c3aed
    class SG,RA,EC derived
```

**Base rates are the point.** Open share ranges **0.17 to 0.91** across users, so
`"opened 8 of 10"` means nothing alone. The context states
`"0.00 vs 0.39 norm (below, 0.0x)"` instead.

---

## 5. Layer 3 — the routing loop

Strict validation, exactly one re-ask with the specific error injected, then a
conservative default. Nothing here repairs a bad response silently.

```mermaid
flowchart TB
    S["build_context"] --> R["render prompt<br/>system 2185 tok + user"]
    R --> C{"cache hit?"}
    C -->|"yes"| P["parse"]
    C -->|"no"| API["Gemini call<br/>temp 0 · structured schema"]
    API --> E{"transport error?"}
    E -->|"429 per-day"| FB["safe_default<br/>digest / unknown / 0.5 / none"]
    E -->|"429 per-minute"| W["honour retry-after"] --> API
    E -->|"no"| P

    P --> V{"validate<br/>12 checks"}
    V -->|"clean"| BD["build_decision<br/>indices → real ids"]
    V -->|"violations"| RA{"already re-asked?"}
    RA -->|"no"| RQ["append the specific<br/>validator errors"] --> API
    RA -->|"yes"| FB

    BD --> OUTR["RouterDecision"]
    FB --> OUTR

    classDef bad fill:#fee2e2,stroke:#b91c1c
    classDef good fill:#dcfce7,stroke:#15803d
    class FB bad
    class BD good
```

The 12 checks: JSON object · exactly five keys · `action` in 3 · `message_type`
in 11 · non-empty reason · confidence a real number in `[0,1]` · every evidence
index drawn from the offered shortlist · at most 2 distinct.

**Measured on the final run: 110/110 valid first try, 0 re-asks, 0 fallbacks.**

---

## 6. Layer 4 — the override guard

Runs after the model and can only hold an action or move it toward less
interruption. `apply.py` **raises** if a rule ever tries to promote a row.

```mermaid
flowchart TB
    IN["RouterDecision"] --> R1{"scam_signature<br/>4-way conjunction"}
    R1 -->|"fires"| M1["mute / scam — TERMINAL"]
    R1 -->|"no"| R2{"reported_sender<br/>unanimous + zero opens"}
    R2 -->|"fires"| M2["mute / spam — TERMINAL"]
    R2 -->|"no"| R3{"opted_out_promotions"}
    R3 -->|"fires"| C1["cap at digest<br/>mute if low-value"]
    R3 -->|"no"| R4
    C1 --> R4{"muted_by_user"}
    R4 -->|"fires"| C2["cap at digest<br/>steps aside for a<br/>direct actionable request"]
    R4 -->|"no"| R5
    C2 --> R5{"quiet_hours"}
    R5 -->|"fires"| C3["notify → digest<br/>unless urgent+addressed<br/>or payment from trusted"]
    R5 -->|"no"| OUT2["final row"]
    C3 --> OUT2
    M1 --> OUT2
    M2 --> OUT2

    OUT2 --> INV{"severity increased?"}
    INV -->|"yes"| ERR["AssertionError<br/>guard may only make things safer"]
    INV -->|"no"| DONE["write + audit"]

    classDef force fill:#fee2e2,stroke:#b91c1c
    classDef cap fill:#fef3c7,stroke:#b45309
    classDef inv fill:#e0e7ff,stroke:#4338ca
    class M1,M2 force
    class C1,C2,C3 cap
    class INV,ERR inv
```

`notify (2) → digest (1) → mute (0)`. Monotonicity is asserted across **110 rows
× 3 starting actions**. When a rule changes the action it also rewrites `reason`
and `confidence`, so a row never states one thing and justifies another.

---

## 7. Evaluation and the regression loop

The labelled data is read here and nowhere else.

```mermaid
flowchart TB
    SMP["sample_messages.csv<br/>30 labelled rows"] --> STRIP["strip_labels<br/>remove the 5 answer columns"]
    STRIP --> PIPE["same pipeline<br/>L2 → L3 → L4"]
    PIPE --> SCORE["metrics.py"]

    SMP --> GOLD["gold answers"] --> SCORE
    SCORE --> A["action / type / joint accuracy"]
    SCORE --> C["confusion matrix<br/>inversions priced 5x, scam 7.5x"]
    SCORE --> E["evidence P / R / F1"]
    SCORE --> K["calibration split<br/>router vs guard rows"]

    A --> LED[("evaluation/runs.csv<br/>append-only ledger")]
    C --> LED
    E --> LED
    K --> LED
    LED --> D["deltas vs previous run<br/>3.33pp noise floor = 1 of 30"]
    D --> T["trade-off warnings<br/>acc ↑ but calibration ↓"]

    STRIP -.-> LEAK["--leak-check<br/>no answer-shaped key<br/>reaches any prompt"]
    PIPE --> GS["golden set — 7 adversarial rows<br/>constraints, not exact labels"]

    classDef guard fill:#e8f5e9,stroke:#2e7d32
    class LEAK,GS guard
```

Latest recorded run: **action 86.7% · type 70.0% · joint 66.7% · ECE 0.081 ·
cost 0.133/row · zero severe errors · golden 6/7.**

---

## 8. The decision this whole system exists for

Two rows with **identical text and the identical attached image**, sent to two
different users, labelled oppositely. Nothing in the content separates them.

```mermaid
flowchart LR
    IMG["Same message text<br/>Same image"] --> A1["sample_msg_044<br/>recipient A"]
    IMG --> B1["sample_msg_045<br/>recipient B"]

    A1 --> AH["opened 8 of 10<br/>0 mutes · 0 reports"]
    B1 --> BH["opened 0 of 6<br/>6 mutes · 1 report"]

    AH --> AR["digest / promotion"]
    BH --> BR["mute / promotion"]

    classDef keep fill:#dcfce7,stroke:#15803d
    classDef drop fill:#fee2e2,stroke:#b91c1c
    class AR keep
    class BR drop
```

Both are in the golden set. **If a change breaks exactly one of them,
personalization has collapsed back into content matching** — which is the failure
this architecture is built to prevent.

---

## Known-wrong, as of this revision

- `rule_reported_sender` overwrites `message_type` with `spam`. On labelled data
  it fired twice and was **wrong both times** (gold was `scam`). It fires **9
  times** on the submission. One-line fix, not yet made.
- `quiet_hours` downgraded two correct `notify`s (`msg_062`, `msg_077`) — admin
  event notices with real deadlines.
- Evidence F1 is **0.327** — the weakest graded axis.
- `evaluation/main.py` never calls `client.save()`, so each eval run re-pays for
  30 model calls.

See the "Known limitations" section of `code/README.md` for the full list.
