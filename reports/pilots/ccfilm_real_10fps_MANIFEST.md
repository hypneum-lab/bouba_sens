# CC-film real-AV bridge cache manifest (ADR-0019 §E1.3d)

- Source film: https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
- License: CC-BY 3.0 (Big Buck Bunny, Blender Foundation 2008)
- N frames: 5965 @ 10.0 fps  (596s of film)
- Film file SHA-256: `f78f39603e6774907f2faafabf26a667f4a6fc31769ec304a8a8f7c62d280508` (verified before extraction)
- Film licence: CC-BY 3.0, (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org
- Audio: mono 16000 Hz, mel-128 (scipy STFT n_fft=1024 hop=256, HTK mel bank from scratch)
- Label classes populated: 4

## Per-channel provenance

- `audio.pt`  : REAL — 128-D log-mel spectrogram of the film soundtrack (the perceptual audio stimulus itself).
- `vision.pt` : REAL — (16,16) per-frame mean-intensity patch grid of decoded movie frames (the perceptual visual stimulus itself).
- `labels.pt` : REAL — shot-cut classes from frame-to-frame difference, quantised to 4 by quantiles.
- `tactile.pt`: SYNTHETIC — RNG(seed=0) stream modulated by the audio energy-flux envelope. A film has no tactile channel; NOT measured signal.
- `gravity.pt`: SYNTHETIC — same construction (3-D).
- `force.pt`  : SYNTHETIC — same construction (6-D).

## SHA256

- `audio.pt`: `4a24683c45e6aefa39166490e0ada3274a0fbfbf80d74a6cc2f850e05266fb46`
- `vision.pt`: `8a6345c18b8432c022df7f9c6d3cfa53b042bc01f6d72726969ea59da19430e2`
- `tactile.pt`: `8dba289494beae9caba2c13566987d899c1728e08a5a836ef7b979f0aa005d48`
- `gravity.pt`: `4e48d64b2544f698aa82e0ae9a41202305d5c27575b5cd8f5feec987794d19d5`
- `force.pt`: `48ab271f91897164e1f674f0fe6c08e00bafeb605f6f6d0b0d18cdc390edef01`
- `labels.pt`: `860a407f13f8c292a75ae2ec34dba8d4817804baaf9a7e58deb2eb5c015bcfda`

## Source download (PI approval 2026-10-28 19:21)

- URL: https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4.zip (the plain .mp4 URL returns 404 since at least 2026-10-28)
- Downloaded 2026-10-28T18:22:16Z on macM1; unpacked with /usr/bin/unzip
- .zip: 64,657,225 bytes, SHA-256 109e3ede8790bd633f374ca311d9cc61dce8d7f98f5b0797ca98199c9fbceedf
- .mp4: 64,657,027 bytes, SHA-256 f78f39603e6774907f2faafabf26a667f4a6fc31769ec304a8a8f7c62d280508
- Licence: CC-BY 3.0, (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org
- Identity check: a 2-fps / 300-s re-extraction from this .mp4 reproduced the six tensor SHA-256 of data/ccfilm_real/MANIFEST.md exactly.
