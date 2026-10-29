# Goldens

Outputs of the upstream Unlimited-OCR 2 reference implementation (the
Hugging Face model code at revision 5632329, PyTorch 2.13, transformers
4.52.1), recorded once outside this repository and committed as data. No
build step and no test here runs Python. `golden_test` checks the C++
engine against them.

Each directory is one page:

| File | What |
|---|---|
| `page.png` | the page raster fed to the reference |
| `letterbox.png` | the reference's 1024 x 1024 global view, as 8-bit pixels |
| `prompt_ids.txt` | the prompt token ids, one per line |
| `tokens.txt` | the generated token ids, one per line, EOS excluded |
| `output_raw.txt` | those tokens decoded, special tokens kept |
| `projector.f32` | the global view's projected features, little-endian fp32, [256, 1280] |
| `sam.f32` | SAM's output for the global view, little-endian fp32, [1, 1024, 16, 16] |
| `first_logits_top20.tsv` | the 20 largest first-step logits (id, value) |
| `meta.txt` | precision, device and versions of the recording run |

`quickstart-p2` is page 2 of gRParse's `quickstart-sample.pdf`, rendered
at 150 dpi, run in base mode with no n-gram ban. The `fp32-cpu` recording
is the exactness gate; `bf16-cuda` is kept beside it to show how far the
deployed precision drifts.
