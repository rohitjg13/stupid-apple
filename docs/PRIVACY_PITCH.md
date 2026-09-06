# The privacy pitch

`docs/DPDP.md` is the compliance write-up. This is what to *say*, and — just as
important — what not to say, because the strongest version of this argument is
the honest one and the overclaimed version collapses under one question.

---

## The one line

> **We never find out who anyone is, because we throw the picture away before we
> start counting.** Everything downstream of the camera sees four numbers per
> person: x, y, width, height.

---

## What YOLO is, accurately

YOLO stands for **You Only Look Once**, and it is worth knowing exactly what that
means, because a judge may know.

It describes the *architecture*. Older detectors (the R-CNN family) work in two
stages: first propose a few thousand candidate regions, then **crop each one** and
run a classifier over that crop, one person at a time. YOLO does not. It divides
the whole image into a grid and, in a **single forward pass**, predicts every box
and every class label simultaneously. One look at the whole frame, one answer.

Three consequences, in descending order of how much they actually matter:

1. **It is fast** — one pass instead of thousands. That is why it runs in real
   time on a Jetson Nano, which is why **no video ever leaves the building**.
   This is the real privacy win, and it is a consequence of the speed.
2. **There is no crop stage.** No point in the pipeline isolates one person and
   re-processes them, because the architecture has no such step.
3. **It is a detector, not a recogniser.** Its entire output vocabulary is a COCO
   class label and a rectangle. It emits `person, (x, y, w, h), 0.87`. It has no
   concept of *which* person and no mechanism to compare one to another.

> **Say this carefully.** "You only look once" is a statement about speed and
> architecture, *not* a privacy guarantee. You could bolt a face recogniser onto
> YOLO's output tomorrow. The privacy does not come from the detector. It comes
> from what we do next — and refuse to do next.

---

## The chain of custody — the argument that actually wins

Follow one frame through the system and watch the person-ness disappear:

| Stage | What exists here | Could you identify anyone? |
|---|---|---|
| Camera → RAM | A frame. A real picture of real people. | Yes — for a few milliseconds |
| Detector | Frame in, `person (x, y, w, h)` out | The output is four integers |
| **Frame discarded** | Buffer reused. Never written to disk. | **The picture is gone** |
| Tracker | Rectangles matched to rectangles by overlap | Nothing but geometry |
| Journeys | Positions over time | "Someone stood here 12 s" |
| Database | Counts, durations, tallies | No |

**Nothing about appearance survives one frame.** Not a crop, not a thumbnail, not
a colour, not an embedding. By the time anything is *stored*, a shopper is a
duration and a coordinate.

---

## The part that proves we meant it

Anyone can claim they don't store faces. Here is a decision that cost us
something:

Shoppers walk behind shelves. When they came out the other side we were counting
them **twice** — inflating footfall, splitting one 30-second dwell into two
15-second ones. The textbook fix is appearance re-identification: embed each
person, match the embeddings across the gap. It is well understood, it is a
library call, and it would have worked immediately.

**We refused it**, because an appearance embedding *is* a biometric identifier —
it is precisely the thing that could recognise the same shopper next Tuesday, or
in another store.

So we solved it with physics instead. When someone reappears, we ask three
questions:

- **Could they have got there?** A person cannot cross the shop in 200 ms.
- **Are they the same size?** People do not change height while hidden.
- **Did they keep going the way they were heading?** Coming back out the side
  they went in is somebody else.

That is the whole re-identifier. It works for a few seconds and a few metres, and
it is **incapable** of recognising anyone tomorrow, in another aisle, or on
another camera — because it has nothing to recognise them *with*. On our test
data it cut identity churn by 27 % (1.73× → 1.26×).

**The privacy property is a consequence of the architecture, not a promise in a
policy document.** We cannot leak biometric data because we never compute any.

---

## Hostile questions, and honest answers

**"Isn't a bounding box personal data?"**
It is data *about* a person, but on its own it identifies nobody — it is a
rectangle and a timestamp. The Act is concerned with data relating to an
*identifiable* individual; we generate no identifier, store no image, and cannot
link a track to a person or to another visit. We're engineers, not lawyers, and
`docs/DPDP.md` is a compliance write-up rather than legal advice.

**"You're still filming people."**
The camera sees, the same as a shop mirror does. The difference is that nothing
is recorded. There is no footage to subpoena, leak, or sell, because there is no
footage.

**"You could just turn re-identification on."**
Yes — and that is exactly why we're showing you the code and the choice rather
than only the outcome. Turning it on would be a visible, reviewable change to a
named module, not a hidden setting.

**"What if the network is compromised?"**
Cloud sync is off by default, and when enabled sends 15-minute aggregate counts.
There is nothing per-person to intercept. The system is fully functional with no
network at all.

**"Why not just use a Raspberry Pi and the cloud?"**
Because then the video leaves the building. On-device inference is what makes
"nothing is stored" achievable rather than aspirational.

---

## Do not say

- ~~"YOLO only looks once, so it's private."~~ Conflates speed with privacy.
  Anyone who knows the architecture will catch it and stop believing the rest.
- ~~"It's anonymous."~~ Say what is true and specific: no images stored, no
  appearance features computed, identifiers reset every restart.
- ~~"It's GDPR/DPDP compliant."~~ Say it is *designed so that most of the Act
  does not apply*, and show why.

---

## Thirty-second version

> It runs a person detector on the device — one pass over the frame, out come
> rectangles. Then we throw the frame away. Everything after that point is
> geometry: rectangles matched to rectangles, positions over time, counts in a
> database. We never compute what anyone looks like. We had a real reason to —
> people vanish behind shelves and we were double-counting them — and we solved
> it with physics instead: you can't teleport, you don't change height, and you
> don't reverse direction the moment you're out of sight. That re-link lasts a
> few seconds and can't recognise you tomorrow, because there's nothing stored to
> recognise you with. Nothing leaves the building either — the whole point of
> running it on the board is that there's no video to upload.
