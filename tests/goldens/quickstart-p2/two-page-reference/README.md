# Two-page decoder reference

Two copies of the known page, one document prompt. Recorded by the C++ fp32
reference decoder at source `5b52adda4e760c03e7bd374d41025811d9c71ee8` on
krick-1, 2026-11-08. The diagnostic uses the existing upstream fp32
`../projector.f32` for each page, with the original checkpoint newline and
separator embeddings. It does not rerun or claim to qualify CPU vision.

`intel_multipage_reference` uses two CPU threads, n-gram size 35, window 1024,
page-count guard, per-page cap 600 and document budget 1203, matching the
two-page RPC configuration. It completed with 348 tokens and EOS. The raw
file excludes closing EOS, matching the existing single-page golden format.

Source, binary, projector and all eight model hashes, full raw output and
exit status are in `docs/receipts/intel-two-page-reference.txt`. Binary SHA256:
`2c2ecc5b24151503269aa08dae9556469deb1cae632e8c53172e6f88dae32fae`.

The original Intel private gate applied the single-page title structure to
both pages and failed. The reference reproduces LEVEL1 on page 1 and LEVEL2
on page 2; this separate oracle preserves that context-dependent structure.
Single-page expectations and the existing 2/1000 geometry tolerance are
unchanged. The original failed receipt remains preserved.
