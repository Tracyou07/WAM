# Third-Party Notices

OpenWAM is distributed under the GNU Affero General Public License v3.0,
subject to the attribution notice in `NOTICE`, except for third-party material
identified below. Those components retain their original copyright and license
terms.

## Prior OpenWAM Contributions

OpenWAM development contributions were previously received and distributed
under the MIT License. Its notice is retained in `LICENSES/MIT.txt`. The public
release as a collective work is distributed under AGPL-3.0-only; rights already
granted for earlier versions are not revoked.

## LingBot-VA

- Component: `src/open_wam/third_party/lingbot/model.py`
- Upstream project: <https://github.com/robbyant/lingbot-va>
- Copyright: 2024-2025 The Robbyant Team Authors
- License: Apache License 2.0
- License text: `LICENSES/Apache-2.0.txt`

The vendored module is adapted to OpenWAM's import and runtime boundaries.
Its upstream attribution and Apache-2.0 terms are preserved.

## Stanford Institutional Marks

- Components: `docs/assets/affiliations/stanford-wordmark.png`,
  `docs/assets/affiliations/stanford-ai-lab.jpg`,
  `docs/assets/affiliations/stanford-svl.png`,
  `docs/assets/affiliations/stanford-stai.png`, and
  `docs/assets/affiliations/stanford-src.webp`
- Sources: [Stanford Identity Guide](https://identity.stanford.edu/visual-identity/stanford-logos/wordmarks/),
  [Stanford Artificial Intelligence Laboratory](https://ai.stanford.edu/logo/),
  [Stanford Vision and Learning Lab](https://svl.stanford.edu/),
  [Stanford Translational AI (STAI) Lab](https://stai.stanford.edu/),
  and [Stanford Robotics Center](https://src.stanford.edu/)
- Rights holder: Stanford University

These unmodified marks identify the project's institutional affiliation. They
are proprietary, are not licensed under AGPL-3.0-only, and may not be reused
except as permitted by Stanford University's applicable brand and trademark
policies.

The GradientWAM distribution omits the institutional image files listed above.
This retained upstream notice does not assert institutional affiliation for
GradientWAM; see `README.md` and `NOTICE` for derivative attribution.

## Diffusers Wan2.2 conversion recipe

- Component: `src/gradientwam/prepare_assets.py` (VAE configuration/key mapping)
- Source reference: [Diffusers 0.37.1 converter](https://github.com/huggingface/diffusers/blob/v0.37.1/scripts/convert_wan_to_diffusers.py)
- License of the source recipe: Apache License 2.0
- License text: `LICENSES/Apache-2.0.txt`

The public converter adapts that recipe to the native OpenWAM frontend layout.
External model weights retain their own upstream terms.

## VRFM and CAGrad method references

The new variational module and two-task CAGrad solver are independent implementations of published methods; no VRFM or CAGrad author source files are vendored. See `CITATION.bib`, `docs/vrfm_cagrad.md`, and `docs/cagrad_reference.md` for primary references, the reviewed official CAGrad implementation, and differences from its examples. The OpenWAM adaptation is not an official reproduction of either paper's experiments. An official VRFM code repository/license has not been independently verified for this delivery.
