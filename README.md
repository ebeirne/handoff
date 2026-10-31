# Handoff Check

Self-hosted acceptance checks for robot training video. A buyer writes down what "usable" means in a YAML spec.
Handoff Check runs on the machine where the data already lives and reports, clip by clip, what passes, what needs a
person to look, and why. It also reports whether the delivery covers what was ordered.

**Status: v0.1 demo.** The checks are tested on synthetic clips and one hour of real footage, and the hand detector
on 10,000 labeled frames. The thresholds are starting points (see "Accuracy" below).

**Findings:** [we checked 6,640 episodes of public robot training data](docs/public_samples_audit_2026-10.md). Published samples are clean (about 1% flagged); the defects live in raw capture.

Beyond checking deliveries, the same tool covers the rest of robot data operations (see "Robot data operations"
below):

| Command | What it does |
|---|---|
| `handoff check` | accept or flag a delivery of video against a spec |
| `handoff fleet` | triage logs from robots (MCAP or LeRobot): keep, review or drop each session, with reasons |
| `handoff repair` | write a re-aligned copy of a LeRobot dataset whose actions lag the state, from a fleet report |
| `handoff watch` | check clips as they land in an upload folder, while the collector can still re-shoot |
| `handoff passport` | seal a delivery with file hashes, consent coverage, licence and acceptance results |
| `handoff verify-card` | confirm a report card was signed by Handoff and hasn't changed since |
| `handoff console` | one page over all of the above |

**Related project:** [rdq](https://github.com/s1g6r/rdq) (Apache-2.0) audits LeRobot datasets and raw video
deliveries: licence and provenance, desync, frozen or swapped cameras, copies, and per-contributor scorecards. It
covers more checks than Handoff does, and its sync check beat ours on our own benchmark (see "Compared with rdq"
below). Handoff focuses on what happens between two parties: one spec, a verdict both sides can reproduce, a sampled
acceptance review, coverage against the order, and repair.

## Quick start

```bash
git clone https://github.com/ebeirne/handoff.git && cd handoff
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[fleet]"
```

**Check your own data:**

```bash
python tools/public_audit.py your-org/your-dataset   # a LeRobot dataset on Hugging Face: one-page report card, tables only
handoff fleet path/to/dataset_or_logs                # a local LeRobot dataset, or a folder of MCAP / ROS 2 logs
```

**Or watch it catch planted defects** in synthetic video clips:

```bash
handoff demo-clips data/demo                 # synthetic clips with known defects
handoff init-spec data/demo/spec.yaml        # an example spec to edit
handoff check data/demo --spec data/demo/spec.yaml --out reports/demo
```

Open `reports/demo/report.html`.

Would you rather have it run for you, or on raw data before it's published? The first check is free:
[get in touch](https://www.linkedin.com/in/ethanbeirne/).

## Run it yourself, or have it done

Everything in this repository is free to run, change and ship (Apache-2.0). What Handoff sells is what the code can't
do by itself: time, judgment, and a result other people will trust.

| | Run it yourself | Have Handoff do it |
|---|---|---|
| Checks on your data | install, write a spec, tune the thresholds to your data, read the reports | results in 2 days, with the spec tuned to your data and to what your buyer accepts |
| A report card buyers trust | a self-check: your own word | **verified**: signed by Handoff after checking the raw data behind the sample too; anyone can confirm it offline with `handoff verify-card` ($500) |
| Accepting a vendor's delivery | decide from the reports yourself | a verdict per episode, what's repairable, and an accepted list you can invoice against ($1,000 per delivery, up to 100 hours) |
| Raw capture, batch by batch | run `handoff check` on each batch | per collector and session, before your own review ($750 a batch; the first one free) |
| iPhone LiDAR catalogs | read the timestamps and depth maps of every clip | a corrected index: true frame rates, rate switches, depth confidence, and where your own index is wrong ($1,000 up to 100 hours) |
| Fixing what's broken | `handoff repair` re-aligns lagged episodes | a week on your recordings: where the data breaks, what can be repaired instead of re-recorded, and a plan ($3,000) |

A verified card can't be produced with this code alone: it carries a signature only Handoff's private key makes, and
changing any number in it breaks the signature.

## What it checks

| Check | Flags a clip when | Parameters |
|---|---|---|
| `readable` | the file can't be opened or decoded (always on) | none |
| `duration` | it's shorter or longer than allowed | `min_s`, `max_s` |
| `fps` | the frame rate is too low | `min` |
| `resolution` | the frame is too small | `min_width`, `min_height` |
| `bitrate` | the file holds too few bits per second for its frame size, the sign of a messaging-app re-encode after capture | `min_mbps_1080p` (scaled by frame size) |
| `black_frames` | too many sampled frames are black | `max_fraction`, `level` |
| `frozen` | too many sampled intervals show no change | `max_fraction`, `min_diff` |
| `blur` | median sharpness (Laplacian variance) is too low | `min_sharpness` |
| `exposure` | too many frames are mostly blown out or crushed (a glare patch is fine) | `max_overexposed_area`, `max_underexposed_area`, `max_bad_frames` |
| `shake` | motion changes direction frame to frame (smooth head turns and pans are fine) | `max_jitter` (pixels at 640 wide) |
| `hands_visible` | hands appear in too few sampled frames | `min_fraction`, `detection_confidence` (needs a hand model, below) |
| `duplicates` | it nearly duplicates an earlier clip in the delivery | `max_hamming` |
| `streams` | recorded streams (video, IMU, actions, gloves) are missing, drop data, start out of sync or stop early. Reads `<clip>.streams.json` with each stream's timestamps | `required`, `max_start_offset_s`, `max_gap_s`, `min_coverage` |
| `frame_timing` | the frame rate changes partway through, or the recorder dropped frames or stalled, from the timestamp kept for every frame (Record3D `.r3d`) | `max_dropped_fraction`, `max_gap_s`, `allow_rate_change` |
| `depth` | too little of the depth map is high confidence, by the sensor's own per-pixel confidence (Record3D `.r3d`; needs `pip install -e .[depth]`) | `min_high_confidence` |

**iPhone LiDAR clips:** `handoff check` also reads Record3D's `.r3d` files (a ZIP of JPEG frames, LiDAR depth and
confidence maps, and per-frame timestamps). The image checks and `hands_visible` run on its frames as on any video.
The frame rate `fps` judges is the one actually recorded, read from the timestamps, not the metadata's (some
Record3D files say 30 fps and record a steady 15). On a public raw iPhone LiDAR dataset, `frame_timing` flags a clip that records
15 fps for 18 seconds and then 30, which plays normally and which no image check can see.

For a whole catalog, Handoff can re-index every clip from the data itself (the rate actually recorded and any switch,
dropped frames, stalls, measured depth confidence) and list where your own index disagrees; see "Run it yourself, or
have it done" above.

Checks run on frames sampled at `sampling.fps` (default 2 per second), downscaled to 640 pixels wide. `shake` also
looks at the two frames right after each sample point, so it measures motion 33 ms apart rather than half a second
apart.

## Verdicts

- **pass:** every check passed.
- **downgrade:** only checks listed under `rules.downgrade_on` failed. The clip is usable, at a lower grade.
- **flag:** some other check failed. A person should look. This is the default for any failed check.
- **reject:** a check listed under `rules.reject_on` failed. By default only `readable` rejects.

Hard rejects have to be written into the spec on purpose. Every failed check says why and **when** (for example
"black frames at 0:12–0:18, 1:05"), so a reviewer can jump straight to the problem and a contributor can fix it.

## Outputs

| File | What it is |
|---|---|
| `report.html` | for people: summary, coverage, spot-check list, every clip with a thumbnail, reasons and timestamps |
| `results.json` | for machines: every check's value, threshold, reason and timestamps, plus cost, coverage and spot-check |
| `accepted.txt` | the clips that passed, one per line |
| `downgraded.txt` | the clips kept at a lower grade |
| `spot_check.txt` | the passed clips a person should review |

`results.json` includes a **fingerprint**: a hash of the spec, the tool version and every verdict. A vendor and a
buyer who run the same spec and tool version on the same data get the same fingerprint, so they can confirm they
agree without sending footage to each other. We tested this on 20 real 1080p clips. Native Windows and the offline
Linux container produced the same fingerprint, with identical check values on every clip.

On a different machine too: a GitHub-hosted Linux server (AMD EPYC) checked the 4-clip demo delivery and the full
hour (20 clips), with inputs confirmed byte-for-byte by SHA-256. Both got the same fingerprint as the laptop, with
identical check values.
- `.github/workflows/repro.yml` repeats the demo every week against `tests/fixtures/repro_demo.json`.
- It runs the hour (`tests/fixtures/repro_hour.json`) on demand.
- `tools/check_repro.py` reports how far any value moved.

**Incomplete runs are never reported as clean.** If the spec asks for a check that can't run, for example a missing
hand model or a missing system library, the report shows an "Incomplete run" banner, `results.json` has
`"complete": false`, the fingerprint differs from a complete run's, and `handoff check` exits with code 3.

### Spot-check

Automated checks narrow the review; a person still confirms acceptance. The report lists which passed clips to
review:

```yaml
acceptance:
  confidence: 0.95       # if none of the sampled clips is bad,
  max_defect_rate: 0.05  # you can be 95% sure fewer than 5% of passed clips are bad
```

That takes 59 clips, from ln(1 − 0.95) / ln(1 − 0.05). For a batch with fewer passed clips than that, the report
says to review them all. The sample is seeded by the fingerprint, so the vendor and the buyer draw the same clips.

### Speed

`--workers N` checks N clips in parallel, and `--workers 0` uses every core but one. The results are identical to a
single-worker run.

It also records a **cost meter**: compute seconds per hour of video. On a laptop CPU (Ryzen 9 7940HS), one hour
of real 1080p, 30 fps egocentric footage took about 7 minutes, hand detection included.

## Coverage against the order

Add a `coverage` block to the spec, with a CSV that labels each clip:

```yaml
coverage:
  metadata: metadata.csv   # relative to the clips folder or the spec
  key: clip                # column holding the clip path
  field: task              # column to count
  targets: {pour: 20, wipe: 20}
```

The report shows passing, flagged and rejected counts per value, and how many passing clips are still missing.

## Hands check (optional)

`hands_visible` uses MediaPipe's Hand Landmarker (Apache-2.0). Install the extra and download the model file once:

```bash
pip install -e ".[hands]"
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
handoff check data/demo --spec data/demo/spec.yaml --hand-model models/hand_landmarker.task
```

Without `--hand-model`, the check is skipped and the report says so.

**Measured weakness:** on 10,000 labeled Egocentric-10K frames, MediaPipe finds a hand in 85% of the frames where
one is visible, at the default `detection_confidence: 0.1`. At MediaPipe's stock 0.5 it's 73%. When it does report a
hand, it's right 99.5% of the time. It finds both hands in 62% of frames that show two. Misses are mostly gloved
hands, finger cots, hands gripping parts and fisheye edges. So a clip with hands always in view measures about 0.85,
and a `hands_visible` flag means "look at this," not proof. Full write-up: `docs/hands_eval_2026-10-30.md`.
To reproduce: `python tools/eval_hands.py <egocentric_10k.parquet> --model models/hand_landmarker.task`.

## Run it offline (Docker)

```bash
docker build -t handoff-check:0.1.0 .
docker run --rm --network none -v /path/to/clips:/data:ro -v /path/to/models:/models:ro -v /path/to/out:/out \
  handoff-check:0.1.0 check /data --spec /data/spec.yaml --hand-model /models/hand_landmarker.task --out /out
```

The container has no network access, runs as a non-root user, and reads the clips read-only.

**Two-party demo:** `powershell -ExecutionPolicy Bypass -File tools\demo_two_party.ps1` checks the same 4 clips at
the same time in two places: the offline container (the vendor's side) and a native install (the buyer's side). Then
it compares the results. On a laptop it takes about 2 minutes. Both sides get fingerprint `e20e8648aea80d79`, with 3
clips passed and 1 flagged. If rdq and the planted SO-100 copy are present, it then repairs that copy from rdq's
findings.

## Robot data operations

### Fleet triage: which robot sessions to keep

```bash
pip install -e ".[fleet]"
handoff init-fleet-spec fleet.yaml                       # an example to edit
handoff fleet logs/ --spec fleet.yaml --out reports/fleet --workers 0   # 0 = all cores but one
handoff fleet path/to/lerobot_dataset                    # LeRobot datasets need no spec
```

`logs/` is a folder of MCAP files (the ROS 2 default log format), one session per file. A subfolder name becomes
the robot's name. Each session gets **keep**, **review** or **drop**, with the reasons and the times they happened.

**Drop:**
- a required stream was never logged
- the session is too short (for a LeRobot dataset without a spec: under a quarter of its median episode, so
  datasets that are a couple of seconds per episode by design aren't dropped wholesale)
- the arm never moves
- the file can't be read at all

**Review:**
- dropouts
- streams logging below their expected rate
- a stream that starts late or stops early
- a state the dataset itself marks as not measured in more than half the frames (`observation.state_valid` = 0, or
  any `<state column>_valid` flag), as datasets built from hand or pose tracking on video do. Those frames are left
  out of the idle and lag checks, since a zero placeholder would look like an arm standing still. A
  `<state column>_hand` or `_id` column (which hand the state follows) works the same way: the jump at a left/right
  switch isn't counted as motion
- mostly idle time
- actions lagging the measured state, compared with the fleet's usual lag (repairable). The limit is 3 frames,
  wider where the per-episode estimate itself varies, so the flags are outliers rather than noise

A LeRobot dataset built from human video often has an action stream and no robot state, or the reverse. Without a
spec, only the streams the dataset records are required; the report says which one is missing, and the checks that
compare state with action don't run. A spec's own `required` list is kept as written.
- a state that is, value for value, the commands from a frame or two earlier: it was filled in, not measured. Set
  `state_from_commands: true` under `sync` when that is on purpose (retargeted human demos have no robot feedback)
- the log ends abruptly (power loss, crash mid-recording): everything before the cut is recovered and checked

Lagged LeRobot episodes can be fixed rather than dropped. `handoff repair` shifts each flagged episode's actions
back to the fleet's usual lag and writes a new copy, leaving the original untouched:

```bash
handoff fleet data/my_dataset --out reports/fleet
handoff repair data/my_dataset --fleet reports/fleet/fleet.json --out data/my_dataset_fixed
```

`REPAIR_LOG.json` lists every shifted episode and records a SHA-256 of the fleet report it came from. The command
refuses a report made on a different dataset.

**Audit a public dataset on Hugging Face** (a separate script, because it downloads; the `handoff` package itself
never touches the network):

```bash
python tools/public_audit.py org/dataset_name          # one or more dataset ids
python tools/public_audit.py --top 25 --skip-done      # the most-downloaded LeRobot datasets published by organizations
```

It downloads only `meta/info.json` and the `data/` tables, never the videos, runs fleet triage, and writes a one-page
`audit.html` per dataset: what it found with example episodes, what it covers and doesn't, the data revision, each
file's SHA-256 and a verdict fingerprint anyone can reproduce. It also writes a `badge.md` snippet for the dataset card,
plus `index.html` and `summary.csv` across everything audited. A cached table from an earlier, larger run is removed,
so the audit reads exactly the files it lists. For datasets with actions and no state, per-hand flags such as
`observation.hands.valid` count a frame as tracked when any hand was found. Visibility stored inside the state vector
(for example `left_hand.*.visible` dimensions) isn't read yet, and the report says so.

The report counts usable hours per robot and lists what goes wrong most often. It reads message timestamps, plus
numeric fields from the streams named in the spec. Those can be JSON-encoded, or ROS 2 messages in CDR as
`ros2 bag record` writes them (for example `sensor_msgs/JointState` `position`). Images and other payloads are never
decoded. If a stream's values can't be read, the report says which stream and why, instead of skipping it silently.

**Compatibility and scale, measured Oct 30, 2026:**
- **Official ROS 2 test recordings:** all 9 MCAP bags recorded by the ROS 2 toolchain (github.com/ros2/rosbag2,
  Apache-2.0) load. They include compressed chunks, split files, and service and action events. A copy cut off
  mid-download was recovered up to the cut: 218 messages on 14 streams.
- **ROS 2 encoding:** the SO-100 fleet demo below, written as CDR `JointState` and `CompressedImage` messages, gives
  exactly the same verdicts and findings as the JSON version.
- **Speed:** eight 10-minute sessions (1.46 GB, with 10 KB camera frames at 30 Hz) took 9.2 s on one process
  (about 160 MB/s) and 2.9 s with `--workers 0` on a 16-thread laptop (about 500 MB/s). A parallel run gives the
  same report as a single-process run.
- **Robot companies' own public datasets:** 9 LeRobot v3 datasets published by 5 robot companies on Hugging Face,
  2,363 episodes and about 23 hours. They cover UR5e, PiPER, Elite CS66, Flexiv Rizon4 and DK1 arms and the Unitree G1
  humanoid, including datasets that split state and action by body part. On data with measured state it raised no
  false alarm. It found one episode whose 20 state and action values never change for 45 seconds. It also flagged a
  dataset whose state is the previous command; that dataset documents this as intended, and the spec can say so.
- **Human-hand data from phone video (a public phone-capture dataset, 116 episodes, Oct 31):** it counts
  the same 9,469 untracked frames of 28,019 (33.8%) that the dataset card documents, and flags the 23 episodes that
  are more than half untracked. Before it read the validity and hand columns, it called those frames idle and raised
  22 lag flags that were artifacts of placeholders and left/right hand switches. Now 92 episodes are kept, 24 go to
  review (23 untracked, 1 borderline lag), and the lag is a steady 1-2 frames on 115 of 116.

The lag check needs no training, so it gives the same answer on every machine.
- **On real SO-100 data with offsets planted in 10 of 50 episodes:** it found all 10 and flagged no clean episode.
  It measured 7 offsets exactly and the other 3 within one frame.
- **On the untouched dataset:** it kept all 50 sessions.

`tools/make_fleet_demo.py` writes 20 SO-100 episodes as MCAP sessions from two robots, with a different fault
planted in each of seven sessions. The joint data is real, and the camera stream is timestamps only. Add `--ros2`
to write them as a ROS 2 bag would.
- Fleet triage kept all 13 clean sessions.
- It caught each of the seven faults with the right verdict and no extra findings.

### Watch: check clips as they're recorded

```bash
handoff watch uploads/ --spec spec.yaml
```

Polls an upload or recording folder. Once a clip stops growing, it checks it and prints one line a collector can act
on, for example `FLAG shaky.avi <- shake: frame-to-frame jitter 58.2px, above 3.0`. Every result also goes to a
JSON-lines log. The cheapest place to catch a bad clip is where it was recorded.

### Passport: a verifiable delivery record

```bash
handoff passport delivery/ --results reports/x/results.json --consent consent.csv --license "CC BY 4.0"
handoff verify-passport passport.json delivery/
```

The passport records:
- every file's SHA-256 and size
- the acceptance fingerprint and verdict counts
- consent coverage (a CSV with `clip`, `consent_id` and an optional `expires` date)
- the licence

The passport is sealed with its own hash. Verifying reports any file that changed, went missing or was added, and
any edit to the passport itself. `handoff passport` exits with code 1 when clips lack consent or their consent has
expired.

### Console: one page over everything

```bash
handoff console --check reports/x/results.json --fleet reports/fleet/fleet.json \
  --repair repaired/REPAIR_LOG.json --passport passport.json --watch watch_log.jsonl --out console.html
```

## Privacy

Handoff Check runs entirely on your machine. It makes no network calls, sends no telemetry, and stores nothing but
the reports you ask for. Footage never leaves your environment.

## Licenses

Handoff Check is Apache-2.0. Every dependency must be on a permissive-license allowlist. To check:

```bash
python tools/check_licenses.py          # core dependencies
python tools/check_licenses.py hands    # including the hands extra
```

Models with non-commercial licenses (MANO-based hand meshes, MASt3R, the Qwen2.5-VL-3B research license) are never
bundled.

## Accuracy

**First real run (Oct 30, 2026):** 20 clips (one hour, 1080p) from factory 051 of the public Egocentric-10K
subset. The first thresholds flagged all 20. Shake measured head motion between frames half a second apart, and
exposure failed on glare off a glossy bench. After fixing both methods, 19 pass and 1 is flagged for low hand
visibility. That's a clean, curated public sample, so it shows false flags went down. It doesn't show that the checks
catch real defects: that needs labeled bad clips.

Every threshold here is a starting point. Before trusting a check on real footage, label a sample of clips by hand
and measure how often each check agrees:

```bash
handoff label-sheet reports/demo/results.json --out labels.csv
# fill in human_verdict (ok / bad) and, for bad clips, human_bad_checks (e.g. "blur;shake")
handoff accuracy reports/demo/results.json labels.csv
```

That prints precision (of the clips flagged, how many were really bad) and recall (of the really bad clips, how many
were caught), overall and per check.

## Handoff Value (Demo 2)

`value/` answers a different question from Check: not "is this clip clean?" but "did this episode make the robot
better?". It runs in its own environment, because the PushT simulator needs OpenCV 4.

```bash
py -3.12 -m venv .venv-value
.venv-value\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-value\Scripts\pip install gym-pusht==0.1.8 pyarrow matplotlib
.venv-value\Scripts\python -m value.experiment --k 96 --workers 4   # about 40 minutes on a laptop GPU
.venv-value\Scripts\python -m value.report                           # chart + docs/value_demo_*.md
```

Data: LeRobot's [PushT keypoints](https://huggingface.co/datasets/lerobot/pusht_keypoints) (Apache-2.0, 206 human
demonstrations), downloaded into `data/pusht_keypoints/`.

Each episode gets two scores:
- **Value:** the datamodels approach. Many small policies are trained on random halves of the data and tested in
  the simulator. An episode's value is the gap in task success between the policies that saw it and the ones that
  didn't.
- **Consistency:** a policy trained without the episode predicts its actions from its observations. The worse the
  prediction, the less internally consistent the episode, as with a sync bug.

The scores are computed on one set of simulator starting positions. Every comparison afterward uses a separate,
held-out set.

**Sync repair** (`python -m value.sync`) targets one failure: actions and observations offset in time, which no
video or timestamp check can see.
1. A policy trained on the other episodes predicts each episode's actions at offsets within ±1 s.
2. Episodes whose best offset sits at least 0.3 s from the batch median are flagged.
3. Flagged episodes are re-aligned.

On PushT with 20% of episodes desynced, it found 30-41 of 41 bad episodes, and repair restored performance to the
level of dropping the true bad ones. That held on two offsets the rule was never tuned on.

On real SO-100 teleoperation data (`lerobot/svla_so100_pickplace`, 50 episodes, 30 fps), with the rule fixed before
running, it found every planted offset of 0.17 s or more, with 0-2 false flags out of 40, and measured each offset
to within one frame. Results and limits: `docs/value_demo_2026-10-30.md`.

Scan your own LeRobot dataset. This needs only torch and pyarrow, not the simulator:

```bash
pip install -e ".[scan]"
python -m value.scan --root path/to/dataset     # fps is read from meta/info.json
```

On a laptop CPU, the 50-episode SO-100 dataset (19,631 frames) took 27 seconds.

For each episode it reports:
- **too short or no motion:** there's nothing to measure, which is a data problem in its own right
- **its offset** from the dataset's typical offset

It flags an episode only when a model-free cross-correlation agrees on the offset, for joint-space data. If offsets
spread too widely for the "most episodes are aligned" assumption, it says **inconclusive** instead of guessing.

Episodes where only one method sees an offset are listed as **needing review**, and aren't repaired.

**Repair:** add `--repair FOLDER` to write a re-aligned copy of the dataset. Only confirmed flags are shifted, the
original is never modified, and `REPAIR_LOG.json` records every change. End to end on a copy of the real SO-100
dataset with offsets planted in 10 episodes:
- 7 were confirmed and repaired, with no clean episode touched.
- Their action error against the true originals fell by 59-100%, and 3 were restored exactly.
- The other 3 were listed for review.

```bash
.venv-value\Scripts\python -m value.scan --root path/to/dataset --fps 30 --repair path/to/repaired_copy
```

**Compared with rdq (Oct 30, 2026).** On the same planted SO-100 copy, rdq 0.3.4's `sync.action_state_lag`, which
needs no training, did better than our scan:

| | rdq 0.3.4 | Handoff scan |
|---|---|---|
| Planted episodes found | 10 of 10 | 10 seen, 7 confirmed by both methods |
| Offset measured exactly | 10 of 10 | 3 of 7 (the rest within 2 frames) |
| Clean episodes flagged | 0 of 40 | 0 of 40 |
| Repeatable across machines | yes (no training) | no: a CPU run on torch 2.15 confirmed 6 and listed 5 for review, including 1 clean episode |

Fed rdq's offsets instead of ours, `value.repair` fixed all 10 planted episodes and changed no clean episode. Each
episode's mean action error against the original fell from 1.4-3.1 to at most 0.0013; what remains is in the edge
frames. Our model-based scan stays as a second opinion, but
we don't lead with it.

To repair from rdq's findings, which come out the same on every machine:

```bash
rdq check path/to/dataset --json rdq.json
python -m value.repair --root path/to/dataset --out path/to/repaired_copy --from-rdq rdq.json
```

`REPAIR_LOG.json` records the rdq version and a SHA-256 of its report, so a second party can confirm the repair came
from the same findings. The repair refuses a report written for a different copy of the dataset.

`python -m value.community_scan` runs the same scan over popular public LeRobot datasets. Across 15 of them (2,068
episodes) it found no confirmed sync bugs, 8 episodes with no usable motion, and 2 datasets it declined to judge:
`docs/lerobot_survey_2026-10-30.md`. `python -m value.inspect` gives a second opinion and a plot for any flagged
episode.

## Tests

```bash
pytest
```

The tests generate clips with known defects (black, blown out, frozen, blurry, shaky, duplicated, unreadable) and
confirm each one is caught for the right reason.

## Roadmap

1. A stronger hand detector, measured against labeled egocentric frames.
2. Per-check accuracy on hand-labeled clips, including real defects.
3. Multi-stream checks on real multi-sensor data. The `streams` check exists and is tested on synthetic timestamps.
   It can't see an offset the logger itself stamped wrong; Handoff Value's consistency score is aimed at that.
4. Authenticity: flag clips where the task looks performed for the camera rather than done.
5. Repair from any detector's offsets (rdq's first), if teams using it ask for that.
