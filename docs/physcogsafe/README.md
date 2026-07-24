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

Scripted safe-reference videos are labeled separately from VLA rollouts and
unchanged-Eb causal replays. Their machine-readable summaries live under
`assets/evidence/`. The canonical L1-B section contains only native-asset
families: B1 is the former B5 ramekin/gripper case, B2 is the former B6 held
cream-cheese/wine-bottle case, and B3 is the former B7 link7/wine-bottle case.
Every family has a four-way Eb/Er/Ec/Safety display. B1/B2 report N=50 formal
statistics; B3 is explicitly labeled as a fully gated N=1 release sample until
its N=50 formal sweep is available.
