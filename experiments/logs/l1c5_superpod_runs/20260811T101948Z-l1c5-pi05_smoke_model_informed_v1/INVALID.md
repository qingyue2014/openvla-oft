# INVALID — L1-C5-MI-v1 pi0.5 smoke attempt

- Superpod job: `515283`
- Commit: `81cd88596e72a0c879f111cb143873175f880920`
- Classification: `validator_bug`
- Failure: the summarizer was given the frozen scene manifest where it
  expected the registered protocol-amendment schema, then stopped with
  `Protocol amendment scenario does not match evaluation`.

All metrics, videos, rollouts, tables, and partial artifacts from this attempt
are invalid as smoke evidence. The scene, serialized states, frozen oracle,
episode count, pairing, seed, model checkpoint, and policy outputs were not
used to change or recalibrate L1-C5-MI-v1.
