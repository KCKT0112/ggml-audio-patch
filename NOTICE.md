# NOTICE

This repository is licensed under the **Mozilla Public License 2.0** (see `LICENSE`),
except for the `patches/` directory, which is dual-licensed **MIT OR Apache-2.0**
(see `patches/LICENSE`) so the diffs remain acceptable to upstream ggml.

The ported operator implementations originate in the upstream projects listed
below. Their original licenses are **not** superseded by this repository's
license: the grants those authors issued are irrevocable, so the portions
derived from their code remain available under their original terms as well.
This notice is provided under MPL-2.0 §3.4 and must be preserved in
redistributions.

## Upstream provenance of the operators

| Operator(s) | Upstream | Upstream license | Where |
|---|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | **Apache-2.0** (Copyright 2026 ShugoAI LLC) | patch 1 |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) (`support-for-tts` branch) | MIT | patch 1 |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) ([Zabis13/ggmlR](https://github.com/Zabis13/ggmlR)) | MIT | patch 1 |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) ([Zabis13/ggmlR](https://github.com/Zabis13/ggmlR)) | MIT | patch 1 |
| `GGML_OP_ADD_LEAKY_RELU` | authored here for [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) | MPL-2.0 | patch 1 |
| `GGML_OP_CONV_DIRECT_1D` (+`_fused`) | authored here for pc-nsf-hifigan.cpp | MPL-2.0 | patch 1 |
| Ten fused ops (`SUPERTONIC_*`, `GRU`, `ZERO_UPSAMPLE`, `CHANNEL_SHUFFLE`, `AFFINE_PRELU`, `SNAKE`) | [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) (`speech` branch) | MIT | patch 2 |
| Vulkan compute backend for `CONV_DIRECT_1D` | authored here | MPL-2.0 | patch 4 |
| Vulkan persistent disk pipeline cache | ported from [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp) | MPL-2.0 | patch 5 |

Base tree: [ggml](https://github.com/ggml-org/ggml) `30bf868` (v0.19.0) — MIT.

## Third-party components

| Component | License | Notes |
|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) | MIT | patch base tree, not vendored |
| Apache License, Version 2.0 | — | full text shipped at `licenses/Apache-2.0.txt` |

## Contributions

Metal integration and the Metal benchmark round (patch 3) were contributed by
**RigoLigo** and are redistributed under MPL-2.0 with the contributor's consent.

## Trademarks

"DiffSinger", "OpenUTAU" and "ggml" are trademarks of their respective owners;
this project is not affiliated with them.
