# L1-C4 pi0.5 smoke video rejection

- Verdict: **FAIL_HUMAN_PI05_SMOKE_VIDEO_INCOMPLETE**
- Review date: 2026-08-10
- Rejected review-artifact manifest SHA-256: `0d806438b6a995253f7f51525f2be2b4ecdec5ce389b31ac605150ce0b17643d`
- Superpod job: `512763`
- Reviewer finding: `视频太短了，都没看到后面cheese下落过程`

The rollout evaluator executed a 60-step post-success stabilization window but
did not append those observations to the encoded review video. It also stopped
that stabilization loop at the first safety violation. These artifacts cannot
authorize formal evaluation. They are superseded by a smoke rerun that must
record the complete stabilization window without changing the frozen scene,
state bundle, intervention, thresholds, safe reference, model, or seeds.
