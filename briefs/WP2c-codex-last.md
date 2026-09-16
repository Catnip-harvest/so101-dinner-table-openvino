Fixed: the collector’s `grip_phase` left a stale jaw target, so `hold()` reopened the grasp. Stepping and phases now delegate to the parent, preserving seeded DART ordering. `episode.py` and tolerances remain unchanged.

Actual acceptance output:

```text
kept=10 attempts=18 yield=55.6%
npz=episode_000001.npz state.shape=(546, 12) action.shape=(546, 12)
episode_000001_top.mp4 frames=546 seconds=10.92
episode_000001_left_wrist.mp4 frames=546 seconds=10.92
episode_000001_right_wrist.mp4 frames=546 seconds=10.92
is_perturbed=[False, False, False, False, False, False, True, False, False, False]
loadback episodes=10 frames=7005
CONVERT_EXIT=0
remote private=True episodes=10 frames=7005 fps=50
chunk probe: complete
```

Manifest fault histogram:

```json
{
  "cup at release: object 16mm off target": 1,
  "cup at release: object 22mm off target": 1,
  "cup pick: jaws 103mm off the cup's axis": 1,
  "cup pick: jaws 20mm off the cup's axis": 1,
  "cup pick: jaws 29mm off the cup's axis": 1,
  "cup pick: jaws 52mm off the cup's axis": 1,
  "cup pick: jaws 66mm off the cup's axis": 1,
  "cup pick: jaws 85mm off the cup's axis": 1,
  "cup pick: jaws closed but the cup is not between them": 3,
  "cup place: jaws closed but the cup is not between them": 9
}
```

[Private dataset](https://huggingface.co/datasets/VietHwang/dinner-table-probe-phys) · [WP2c report](<C:/Users/vieth/Documents/lablab hackathon/briefs/WP2c-REPORT.md>)

Completed before 20:30 Hanoi. No `.env*` files opened; nothing public.