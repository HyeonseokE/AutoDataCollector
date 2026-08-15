# Table 2 — Dataset list (episode-count splits)

HF dataset paths for each task, split into 20/40/60/80/100 episodes. Method order: **teleop → CaP → Ours**.

> **Split-rule note:** `20/40/60/80` subsets for **teleop** and **CaP** use the *seed-balanced* rule (`keep idx%5 < N/20`, first k of each 5-episode seed block). **Ours** subsets use the *phase-split* rule (phase1=first 80 + phase2=last 20, front-contiguous, 3:1 → 15/5, 30/10, 45/15, 60/20). These two rules differ.

> **`100` rows** point to the original full source repos (mixed namespaces `skkuprism/`, `CoRL2026-CSI/`); all `20/40/60/80` subsets live under `HyeonseokE/`.

> Caveats: `CaP open_pot` source = 115 ep (subsets use first 100). `Ours Closelid` source = 101 ep (first 100 used). `Ours FoldTowel` source = 99 ep (its 80epi = 61+19).


## T1 — pick_place

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/teleop_pnp_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/teleop_pnp_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/teleop_pnp_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/teleop_pnp_80epi
- 100: https://huggingface.co/datasets/skkuprism/teleop_pnp_100ep

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_pnp_baseline_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_pnp_baseline_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_pnp_baseline_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_pnp_baseline_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_pnp_100ep_baseline_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-PickandPlace_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-PickandPlace_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-PickandPlace_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-PickandPlace_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-PickandPlace_Ours_100epi


## T2 — distribute

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_distribute_chocolate_pie_each_plate_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_distribute_chocolate_pie_each_plate_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_distribute_chocolate_pie_each_plate_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_distribute_chocolate_pie_each_plate_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_distribute_chocolate_pie_each_plate_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_distribute_choco_baseline_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_distribute_choco_baseline_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_distribute_choco_baseline_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_distribute_choco_baseline_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_distribute_choco_100ep_baseline_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-DistributeChoco_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-DistributeChoco_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-DistributeChoco_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-DistributeChoco_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-DistributeChoco_Ours_100epi


## T3 — sort

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_sort_blocks_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_sort_blocks_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_sort_blocks_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_sort_blocks_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_sort_blocks_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_sort_RGBblock_to_matchingplate_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_sort_RGBblock_to_matchingplate_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_sort_RGBblock_to_matchingplate_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_sort_RGBblock_to_matchingplate_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_sort_RGBblock_to_matchingplate_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-SortBlock_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-SortBlock_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-SortBlock_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-SortBlock_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-SortBlock_Ours_100epi


## T4 — stack

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_stack_RGBblock_on_bluedish_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_stack_RGBblock_on_bluedish_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_stack_RGBblock_on_bluedish_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_stack_RGBblock_on_bluedish_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_stack_RGBblock_on_bluedish_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_stack_RGBblock_on_bluedish_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_stack_RGBblock_on_bluedish_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_stack_RGBblock_on_bluedish_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_stack_RGBblock_on_bluedish_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-StackBlock_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-StackBlock_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-StackBlock_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-StackBlock_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-StackBlock_Ours_100epi


## T5 — close_pot / close_lid

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_lid_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_lid_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_lid_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_lid_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101_teleop_close_lid_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_pot_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_pot_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_pot_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_pot_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_close_pot_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-Closelid_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-Closelid_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-Closelid_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-Closelid_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-Closelid_Ours_100epi


## T6 — open_drawer

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-Teleop-Open_drawer_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-Teleop-Open_drawer_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-Teleop-Open_drawer_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-Teleop-Open_drawer_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-Teleop-Open_drawer_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_topdrawer_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_topdrawer_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_topdrawer_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_topdrawer_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_open_topdrawer_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-OpenDrawer_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-OpenDrawer_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-OpenDrawer_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-OpenDrawer_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-OpenDrawer_Ours_100epi


## T7 — place_spoon

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_place_spoon_between_bread_and_cereal_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_place_spoon_between_bread_and_cereal_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_place_spoon_between_bread_and_cereal_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-teleop_place_spoon_between_bread_and_cereal_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_place_spoon_between_bread_and_cereal_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_place_spoon_between_bread_and_cereal_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_place_spoon_between_bread_and_cereal_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_place_spoon_between_bread_and_cereal_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_place_spoon_between_bread_and_cereal_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_place_spoon_between_bread_and_cereal_100epi

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-PlaceSpoon_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-PlaceSpoon_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-PlaceSpoon_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-PlaceSpoon_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-PlaceSpoon_Ours_100epi


## T8 — fold_towel (bi-arm)

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-DualArm-teleop_fold_towel_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-DualArm-teleop_fold_towel_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-DualArm-teleop_fold_towel_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-DualArm-teleop_fold_towel_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-DualArm-teleop_fold_towel_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_fold_towel_10fps_stride3_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_fold_towel_10fps_stride3_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_fold_towel_10fps_stride3_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_fold_towel_10fps_stride3_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_fold_towel_10fps_stride3

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-FoldTowel_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-FoldTowel_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-FoldTowel_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-FoldTowel_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-FoldTowel_Ours_100epi


## T9 — open_lid / open_pot

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_open_lid_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_open_lid_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_open_lid_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_open_lid_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101_teleop_open_lid_100epi

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_pot_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_pot_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_pot_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_open_pot_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_open_pot_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-Openlid_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-Openlid_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-Openlid_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-Openlid_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-Openlid_Ours_100epi


## T10 — close_drawer

**teleop**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_drawer_wristfix_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_drawer_wristfix_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_drawer_wristfix_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101_teleop_close_drawer_wristfix_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101_teleop_close_drawer_100epi_wristfix

**CaP**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_drawer_10fps_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_drawer_10fps_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_drawer_10fps_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-cap_close_drawer_10fps_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_close_drawer_10fps

**Ours**
- 20: https://huggingface.co/datasets/HyeonseokE/SO101-CloseDrawer_Ours_20epi
- 40: https://huggingface.co/datasets/HyeonseokE/SO101-CloseDrawer_Ours_40epi
- 60: https://huggingface.co/datasets/HyeonseokE/SO101-CloseDrawer_Ours_60epi
- 80: https://huggingface.co/datasets/HyeonseokE/SO101-CloseDrawer_Ours_80epi
- 100: https://huggingface.co/datasets/CoRL2026-CSI/SO101-CloseDrawer_Ours_100epi
