# How often does MediaPipe see the hands in factory egocentric video?

**Short answer:** at its default setting, MediaPipe's hand landmarker misses about a quarter of frames where the
camera wearer's hands are visible, and finds both hands less than half the time when both are in view. Lowering its
detection threshold to 0.1 recovers a good share of that and barely adds false alarms.

## Setup

- **Data:** Build AI's [Egocentric-10K evaluation set](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation)
  (Apache-2.0). 10,000 frames sampled at random from Egocentric-10K (head-mounted, fisheye, factory work).
- **Labels:** each frame's count of the wearer's visible hands (0, 1 or 2), from gemini-2.5-flash with Build AI's
  published prompt. These are model labels, not human ones, so the numbers below are **agreement with Gemini**. In a
  visual check of 12 frames where Gemini says two hands and MediaPipe finds none, Gemini was clearly right in nearly
  all of them.
- **Detector:** MediaPipe Hand Landmarker (`hand_landmarker.task`, float16), up to 2 hands, frames downscaled to 640
  px wide (as in `handoff check`). Detection and presence confidence set together.
- **Script:** `tools/eval_hands.py`. It runs on a laptop CPU (Ryzen 9 7940HS, 6 workers) in a few minutes per
  setting.

## Results

Gemini labels a visible hand in **96.4%** of frames.

| Confidence | Detector says "hand" | Recall: hand found when Gemini sees one | Precision: Gemini agrees when detector says hand | Both hands found when Gemini sees two | Exact count matches |
|---|---|---|---|---|---|
| 0.5 (MediaPipe default) | 71.0% | 73.4% | 99.8% | 42.6% | 44.6% |
| 0.3 | 75.8% | 78.4% | 99.7% | 50.3% | 51.1% |
| **0.1** | 82.7% | **85.3%** | 99.5% | 61.9% | 60.9% |

Recall during active work (Build AI's `active_labor = yes` frames): 75.1% at 0.5, 80.0% at 0.3, 86.7% at 0.1.

**Confusion at 0.1** (rows: Gemini's count; columns: MediaPipe's count)

| Gemini \ MediaPipe | 0 | 1 | 2 |
|---|---|---|---|
| 0 (358 frames) | 317 | 35 | 6 |
| 1 (2,008 frames) | 696 | 1,043 | 269 |
| 2 (7,634 frames) | 721 | 2,187 | 4,726 |

## What the misses look like

The frames MediaPipe misses most often:
- white cleanroom gloves and orange finger cots
- hands gripping tools or parts, where the palm is hidden
- hands distorted at the edges of the fisheye frame
- blue sleeves covering the wrists

## What it means for Handoff Check

- `hands_visible` now defaults to `detection_confidence: 0.1`.
- A clip with the wearer's hands always in view measures roughly **0.85, not 1.0**. Set `min_fraction` with that in
  mind. The example spec uses 0.5.
- Treat a `hands_visible` flag as "look at this," never as proof. A stronger detector, measured the same way, is the
  next step.
- MediaPipe almost never reports a hand that isn't there, so when it does find hands, that's reliable.
