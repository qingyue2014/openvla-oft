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
example uses the serialized Er state, the policy's 256×256 `agentview`, and the
same 7-D OSC interface as evaluation.
