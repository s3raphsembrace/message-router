# Analysis — design evidence

These two scripts are the evidence trail behind the routing design. They read only
`dataset/`, take no arguments, need no API key, and print their findings. Every
threshold and precedence decision in the pipeline traces back to one of them.

```bash
python code/analysis/collision_check.py
python code/analysis/rule_validation.py
```

## `collision_check.py`

Confirms that personalization is decidable from the provided data — that two
identical-looking messages really do require opposite routing, and which file
resolves each case.

- **Same poster, opposite routing.** Two promo pairs share the *same image file*
  across different users, split cleanly by `user_business_history`
  (`allows_promotions` + `promotions_opted_out_at` + opened/dismissed ratio).
  Two other identical-text pairs do **not** diverge — both recipients have live
  transactional relationships — so duplicate text is not itself a routing signal.
- **Payment reminder, legit vs scam.** Money-related business messages separate
  almost linearly on `official_domain != domain_used_by_sender` combined with
  account age and report count.
- **Muted group, urgent mention.** 14 group messages target groups the user has
  muted. There is no user-name field anywhere in the dataset — mentions are
  literal `@user_id` tokens, and every mention in `messages.csv` addresses the
  recipient, so direct-address detection is exact string matching.

## `rule_validation.py`

Tests four candidate hard rules against the 30 labelled samples. **A rule that
would overturn a provided label is not a hard rule.**

| rule | verdict |
|---|---|
| verified opt-out → mute | **holds** — gate on promotional *content*, not on business-sender |
| reported/muted sender → mute | **fails** — fires on 59/110 and contradicts two labels |
| quiet hours → downgrade | **inert** — 0 labelled samples inside; changes no decision |
| clear scam signature | **holds as a conjunction only** — mismatch alone is not scam |

The two failures are the load-bearing results:

- **Reported/muted sender.** `sample_msg_001` is labelled `notify/urgent` with
  1 report and 2 mutes against that group — but **19 of 21 opened**. The
  discriminator is the open *ratio*, not whether a report ever happened. Only the
  degenerate case (`reported == n AND opened == 0`) survives as a rule; the rest
  becomes a Layer 2 feature.
- **Quiet hours.** Zero labelled samples fall inside a DND window, so the rule has
  no ground-truth support in either direction. Only 8 of 110 live messages are
  affected, and every one already routes to `digest` or `mute` on content alone —
  two OTP scams, an unverified 38-report voice note, a next-day FedEx window, a
  fire-alarm notice, a review request, a school circular, and a personal message
  that says *"No need to reply."* The rule changes nothing and can only ever
  demote a correct `notify`.

Together these produce the guard's operating principle: **a rule belongs in
Layer 4 only if it decides on a fact the model cannot verify, and it can only ever
be right.** Judgments on continuous variables are Layer 2 features.
