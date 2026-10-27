# Source and asset attribution

- This repository derives from https://github.com/om-ai-lab/OmTrackVLA at
  `e9cb1fbd57f8dbcf98c460914251cbcbe20f2f57`. This repository starts with a clean
  import; upstream commit history remains available in the original repository.
  Original source attribution is retained. Its README links Apache-2.0, but that upstream
  snapshot has no root license file; this deployment does not invent or replace
  an upstream license grant. Consult the upstream authors for licensing clarity.
- Bundled modified Habitat-Lab derives from Meta/Facebook Research Habitat-Lab.
  Preserve its source headers and `habitat-lab/LICENSE` (MIT).
- The Spot robot assets retain `data/robots/hab_spot_arm/LICENSE.txt`.
- Pretrained OmTrackVLA, Qwen, DINOv3, SiGLIP and optional Grounding DINO/YOLO
  weights are downloaded separately from their publishers and retain their
  individual model licenses and access conditions. DINOv3 access must be
  approved for the downloading account. No weights or account credentials are
  published here.
- HM3D, MP3D, humanoid and additional simulator assets require separate access
  under their providers' licenses. Private lab recordings are excluded.

The local camera preview, configuration, packaging and benchmark tools are
integration work. They do not change the origin or attribution of the released
OmTrackVLA model and do not imply verified physical-following accuracy.
