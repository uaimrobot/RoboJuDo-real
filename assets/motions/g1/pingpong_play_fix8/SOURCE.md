Copied 2026-11-04 from pingpong logs/staged_fix8/stage4_robust_gait/2026-11-04_13-00-26_stage4-fix8-7 (model_16000.pt + params/): stage4-fix8-7, recipe staged_fix8.
Needs the matching pingpong files: data/planner_tables/targets_mixfix_aim_mixspin_mixe.npz (its training table) and the new configs/hardware.
Select it with: python scripts/run_pingpong.py -c g1_pingpong_real --log --checkpoint assets/motions/g1/pingpong_play_fix8/model_16000.pt
