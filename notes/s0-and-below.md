# S0 and below: what runs before the decision model

An S1 model (Clef, Clairic) is fast, but it's still a network call that reads one input. An LLM is slower and costs
more. Most of the time, nothing interesting is happening, and neither should run.

**S0** is the layer that decides *when* to call S1, and *what* to send it. It runs where the signal is (a browser
tab, a device, a log tailer), on every frame or every event, for free.

```
S-1  signal hygiene       is the input usable at all?            per sample, ~0 cost
S0   on-device sensing    did something change? what, roughly?   per frame/event, local
S1   decision model       what does it mean, typed, with odds    per call, ~40-200 ms
S2   LLM                  open-ended work, checked by S1 + code  per call, seconds
```

## S0 in the wild: two examples

**[Glance](https://github.com/acoyfellow/glance)** (Orb view). MediaPipe Tasks Vision `FaceLandmarker` and
`HandLandmarker` run in the page on every new video frame (it skips frames where `currentTime` hasn't moved).
Blendshapes and landmarks become a handful of continuous signals: smile, stillness, head nod, head shake, head
direction, mouth open, open or closed hand. Those drive the Orb directly. Nothing leaves the page.

**[read-the-room](https://github.com/acoyfellow/read-the-room)**. Browser code does the cheap work before every model
call: frames are downscaled to 512 px JPEG, audio is captured at 16 kHz mono and sent in chunks, and code applies
thresholds after Clef answers (a charades win needs p >= 0.6 on two frames in a row). An early version also counted
people with an in-browser pose model and logged that count next to Clairic's count of the same frame. That was cut
from the booth (28 MB of pose files), but the pairing is the interesting part: **S0 and S1 cross-checking each other**.

## What S0 is good for

| Job | Example | Saves |
|---|---|---|
| **Gate:** call S1 only when something changed | motion, a face appears, speech starts, a log line matches | most calls |
| **Crop:** send only the part that matters | the hand, the face, the region that moved, the last 40 log lines | tokens, latency |
| **Debounce:** require a signal to hold | "smile for 400 ms", "two frames agree" | false triggers |
| **Pre-label:** give S1 hints as state | `{"faces": 2, "hand": "open", "still_ms": 1800}` | S1 accuracy |
| **Cross-check:** compare S0 and S1 on the same input | pose count vs Clairic count; disagreement = ask | silent errors |
| **Fallback:** keep working offline | gestures still move the Orb with no network | availability |
| **Privacy:** raw media never leaves | only landmarks or a crop go up | exposure |

## Below S0

**S-1: is the input usable?** Before anything detects meaning, check that there's a signal: exposure and blur on a
frame, clipping and silence on audio, a camera that's frozen (same frame twice), a tab that's hidden, a stream that's
stale. Bad input sent to S1 gives confident nonsense.

**S-2: is anyone there, and should we look at all?** Presence and consent: the page is visible, the user opted in,
the device is on battery or not, it's a quiet hour. This is policy, not perception, but it belongs below everything.

## Rules of thumb

- S0 never makes the final call on anything that matters. It decides whether to ask, and what to send.
- Every S0 signal should be a bounded number with a known range, so a threshold or a small circuit can use it.
  (This is where exact, hand-written programs like [Handwired](https://github.com/chenglou/handwired) fit.)
- Log S0's decision next to S1's answer. Disagreement is the cheapest error detector you have.
- Measure what S0 saves: calls skipped, tokens cut, false triggers suppressed.
