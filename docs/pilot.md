# Handoff pilot: one real delivery, checked the same way by both sides

**What it answers:** for one delivery of robot training data, which episodes should be accepted and billed, why the
others weren't, and whether the buyer and the vendor would reach the same answer independently.

Contracts for robot data increasingly say "rejected episodes aren't billed" or cap the rejection rate. But each side
decides what "rejected" means with its own tools. The pilot tests whether one shared check settles that.

## How it works

1. **Day 1 (30 minutes, both sides if possible):** we write your acceptance rules as a Handoff spec. They cover what
   counts as usable for this task, such as hands in view, duration, frame rate, sync between streams, coverage of
   tasks and scenes, and which failures flag, downgrade or reject.
2. **Days 2-4:** each side runs Handoff on the same delivery, **on its own machines**. It ships as a Docker container
   that runs with networking turned off. Footage never leaves your machines, and we never see it unless you choose to
   show us.
3. **Day 5 (45 minutes):** we compare and walk through the results:
   - whether both sides got the same fingerprint, which means the same verdict on every episode
   - pass / downgrade / flag / reject for every clip, with the reason and the timestamp
   - the accepted list and the rejection rate, set against what your contract allows
   - coverage against what was ordered
   - a spot-check sample sized for the confidence you want, drawn identically on both sides
   - how often each check agreed with your own reviewers, on a sample you label

For teleoperation data, episodes whose actions and observations are out of sync can be **repaired instead of
rejected**. Handoff writes a re-aligned copy and logs every change; the original is never touched.

If the other side isn't ready to join, you can run the pilot alone first. The spec is a file you can share with them
later.

## What you get

- The reports and results files, yours to keep.
- The spec, reusable on every later delivery and shareable with the other side.
- A measured comparison: your current review time per hour of data, against review with Handoff's flags.
- Where your two sides would have disagreed, and why.

## What we need from you

- One delivery you've already received or sent: 10-50 hours of video, or a LeRobot dataset (more is fine).
- One person who knows the acceptance criteria, for about two hours across the week.
- Your honest feedback, including "this isn't useful."

## Cost

The first three pilots are free. After that, pilots are paid, and we'll agree a price up front.

## Fine print, in plain words

- The software is open source (Apache-2.0). You can read every line before running it.
- It makes no network calls and sends no telemetry.
- Handoff reports what it measured. It doesn't certify anything as compliant or lawful, and it doesn't hold money.
- Thresholds are starting points. Part of the pilot is measuring how well they fit your data.

*A written pilot agreement will be reviewed by counsel before anything is signed.*

Contact: {email}
