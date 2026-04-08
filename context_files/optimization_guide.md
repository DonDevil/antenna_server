# Antenna Optimization Guide v1

Use this guide when selecting the next refinement action from CST feedback.

## Primary objective priority
1. S11 / return loss / VSWR matching
2. center-frequency alignment
3. bandwidth expansion
4. gain and efficiency improvement

## Heuristic mapping
- Resonance above target frequency -> increase effective patch length.
- Resonance below target frequency -> decrease effective patch length.
- Poor S11 / high VSWR -> adjust feed width, feed offset, or coupling before changing the whole radiator.
- Bandwidth too narrow -> increase substrate height, widen patch/feed, or use stronger coupling.
- Gain too low -> increase effective radiating aperture and substrate support area.

## Secondary targets
- `Zin ≈ 50 + j0` indicates the matching network is on target.
- `AxialRatio < 3 dB` is mainly relevant for circular polarization designs.
- `SLL minimize` and `F/B maximize` should only influence action choice after the primary metrics are under control.

## Action-selection rule
Choose the safest action from the candidate list only. Never invent new actions. Prefer targeted matching corrections before broad geometry edits.
