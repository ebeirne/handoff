# We checked 6,640 episodes of public robot training data. The problems aren't where you'd expect.

*October 31, 2026. No company is named on purpose: these are their free samples, and the point is the pattern.*

Robot labs now buy demonstration data by the thousand hours: teleoperation, handheld grippers, first-person video
from phones and headsets. Most vendors publish a free sample on Hugging Face so buyers can see what they'd get. We
wanted to know how those samples hold up under the checks a buyer would run on a delivery, so we built them and ran
them.

## What we checked

14 public datasets from robot-data vendors and robot makers: **6,640 episodes, 30.8 hours.** They cover
robot arms (PiPER, SO-101, DK1, Elite CS66), a Unitree G1 humanoid, handheld UMI-style grippers, and
human-hand data captured from phones and headsets. Another 16 we couldn't read: 6 are gated (samples on request),
7 had moved or disappeared, and 3 were too large for a quick pass or in a format we don't read yet.

Every episode went through the same checks Handoff runs on a delivery: missing or late streams, dropouts, frozen or
idle recordings, actions out of step with the measured state, recordings too short to use, and how much of each
episode has a tracked hand.

## What we found

**1. Published samples are clean.** Leaving out flags that are there by design, **65 of 6,640 episodes (1%) were
flagged**. About a third come from one dataset (point 3); the rest are mostly idle stretches or small lags a
person should glance at. One stood out: a bimanual
recording where all 20 state and action values never change for 45 seconds. Either the arms sat still for the whole
take or the logging froze. In a three-hour dataset, that's easy to miss by eye.

**2. "By design" is a real category, and buyers need to know it.** In two datasets, the robot's "state" is a copy of
the commands, value for value: they're human demonstrations retargeted to a robot, so no robot ever
measured anything. One dataset documents this on its card; a buyer who doesn't read the card would train on it as if
it were proprioception. On a humanoid, the legs log their measured position as the "action" because a balance
controller drives them. If you measure action/state lag across all joints, those legs pin it at zero and hide a
lagged arm. Handoff now reports these instead of mistaking them for faults, and a spec can mark them as intended.

**3. The defects live in raw capture.** The single dataset published close to raw is the one with real findings:
about a third of its frames have no tracked hand, and 23 of its 116 episodes are more than half untracked. That
matches what a large data producer told us about their own pipeline: the biggest net is checking at capture time,
while the collector can still re-shoot, because the worst defects (dropped frames, desynced or missing streams,
incomplete episodes) can't be fixed downstream. One industry cost guide puts the share of teleop episodes
discarded in quality filtering at 20-30%.

So a free audit of someone's published sample mostly confirms they curate well. The interesting question is what
gets thrown away before publishing, and what that costs the vendor who's paid per accepted episode.

## What Handoff is

An open-source (Apache-2.0) quality layer for robot training data that runs **on the data owner's machines, with
networking off**. Footage never has to leave the building.

- `handoff check`: accept or flag a video delivery against a written spec (hands in view, duration, frame rate,
  stream sync, coverage), with the reason and timestamp for every clip.
- `handoff fleet`: triage MCAP/ROS 2 logs and LeRobot datasets into keep, review or drop. On LeRobot's real SO-100
  recordings with action offsets planted in 10 of 50 episodes, it found all 10 and flagged none of the 40 clean ones. It reads
  1.46 GB of logs in 2.9 s on a laptop.
- `handoff repair`: episodes whose actions and observations are offset in time get a re-aligned copy instead of
  being thrown away. On PushT with a hidden 0.4 s offset in 20% of episodes, repairing the flagged ones matched
  dropping the truly bad ones (57% vs 56% coverage).
- `handoff watch`: checks each clip as it lands, while the collector can still re-shoot.
- `handoff passport`: a sealed record of a delivery (file hashes, consent, licence, verdicts).

The part we care most about: **buyer and vendor run the same spec on the same data and get the same fingerprint**,
on different machines. Acceptance stops being one side's word against the other's.

## What it doesn't do (yet)

It doesn't judge whether a task was done well or a demonstration was staged for the camera; those are semantic calls
that still need people or task-specific models. Its hand detector misses gloved hands and hands gripping handles.
And for auditing published LeRobot datasets, [rdq](https://github.com/s1g6r/rdq) is further along than we are and
worth using; Handoff's repair can run from rdq's findings.

## Try it, or let us try it on yours

The code, tests and every benchmark are in this repository. Benchmarks: [the README](../README.md) and [the repair study](value_demo_2026-10-30.md). To get a report card for your own public
LeRobot dataset (it downloads only the tables, never the videos):

```
python tools/public_audit.py your-org/your-dataset
```

If you collect or buy robot data and
throw part of it away, we'd like to run Handoff on one raw, unfiltered batch, on your machines, and measure how much
of the discard it catches early or repairs. The first three pilots are free.

Ethan Beirne, New York ([LinkedIn](https://www.linkedin.com/in/ethanbeirne/)). I'm looking for a robot-learning co-founder: if this is your problem too, get in touch there or open an issue.
