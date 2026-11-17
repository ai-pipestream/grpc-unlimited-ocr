# Metal implementation and qualification status

The full native Metal vision and decoder backend is implemented. Build and
configuration instructions are in [metal.md](metal.md). It uses native
Metal kernels and MPS only for dense matrix multiplication; it does not
replace the forward pass with an inference runtime. CPU fallback is not
automatic.

The earlier primitive-only checkpoint is superseded. On the Apple M2,
source `2db7d72a473b7f064ea3ec3cc97ccf6a497b420a` passed the full typed-document
TCP matrix: six alternating one/two-page requests and three cancellation,
admission-release and recovery cycles. Content, geometry, page identity,
references and repeated serialized Document payloads were checked. Exact
source, model and binary identities and raw logs are in the
[typed-document receipt](receipts/metal-m2-typed-documents-20261108/README.md).

Those results qualify the recorded source and fixtures. They are not a
claim that every subsequent combined branch or release artifact has been
tested. Final combined-source qualification is tracked in
[gRParse #85](https://git.rokkon.com/ai-pipestream/gRParse/issues/85).
Performance work and broader document/long-running coverage remain separate
from the completed fixture correctness gates.
