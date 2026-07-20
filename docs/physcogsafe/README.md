# PhysCogTest project overview

Open `index.html` directly in a browser, or serve this directory locally:

```bash
python3 -m http.server 8000 --directory docs/physcogsafe
```

Then visit `http://localhost:8000`.

The page has no network dependencies. Embedded videos live under
`assets/videos/` and use relative paths, so the complete directory can be
copied to another machine or deployed as a static site.

Video provenance is written next to every player. Formal, partial, smoke, and
same-configuration preview assets must remain visibly distinguished when the
page is updated.

Scripted safe-reference videos are also labeled separately from VLA rollouts.
Their machine-readable run summaries live under `assets/evidence/`; the B1
matched pair uses the same serialized Er `demo_1`, BDDL, seed, 10-step wait,
and policy 256×256 `agentview` on both sides. B3 and B4 each have their own
strict matched pair using that family's Er `demo_0` under the same controls;
B1 task-6, B3 task-6, and B4 goal-layout remain under separate headings and
are never paired across families. The B3 VLA video is explicitly labeled as a
deterministic replay of the formal initial state because the original
visibility-corrected 50-state run did not save MP4 files. The scripted side
uses the same 7-D OSC interface as evaluation but remains explicitly labeled
as a controller reference. The L1-B family overview now gives each B1/B2/B3/B4
scene a four-way Eb/Er/Ec/Safety display; the four conditions are not collapsed
into one aggregate video.
