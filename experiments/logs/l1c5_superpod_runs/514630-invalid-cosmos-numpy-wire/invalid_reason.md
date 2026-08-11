# Invalid L1-C5 Cosmos smoke run: NumPy wire incompatibility

- Superpod job: `514630`
- Bound commit: `5ef6069163d7591091a4c32ead6233003a31f760`
- Classification: invalid evaluator/inference transport failure; no model outcome

The isolated Cosmos policy process loaded the official checkpoint and the
frozen LIBERO evaluator passed its native-only and exact asset-inventory gates.
However, every episode stopped before its first valid policy-phase step when
the NumPy 1.26 evaluator attempted to unpickle an action array produced by the
NumPy 2.2 Cosmos process:

```text
Episode error: No module named 'numpy._core.numeric'
```

The generated trajectories and very short videos contain no valid policy
execution and must not be interpreted, reviewed, or included in metrics.  No
formal Cosmos job was launched from this run.  The replacement transport
encodes arrays as dtype/shape/raw-byte records containing only Python built-in
types, preserving the isolated official Cosmos runtime and the frozen LIBERO
simulator stack without relaxing any experiment gate.
