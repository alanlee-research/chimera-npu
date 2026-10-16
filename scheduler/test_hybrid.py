"""Sanity tests and a small simulation of the reference scheduler (python -m pytest scheduler)."""
from hybrid import AimdParams, AimdScheduler, keep_tile_detection, merge_frame


def test_initial_rate_and_tiles_per_frame():
    s = AimdScheduler(AimdParams(frame_period_ms=1000 / 30, job_ms=11.2, num_tiles=4))
    assert abs(s.mu_max - 2.976) < 0.01 and abs(s.mu - 0.85 * s.mu_max) < 1e-9
    tiles = [sum(j[0] == "tile" for j in s.plan_frame()) for _ in range(30)]
    assert all(t in (1, 2) for t in tiles)                    # one full-frame job plus 1-2 tiles
    assert abs(sum(tiles) / 30 - (s.mu - 1)) < 0.1           # fractional rate realized on average


def test_round_robin_and_aimd():
    s = AimdScheduler(AimdParams(frame_period_ms=1000 / 30, job_ms=11.2, num_tiles=4))
    seen = [j[1] for _ in range(10) for j in s.plan_frame() if j[0] == "tile"]
    assert seen[:4] == [0, 1, 2, 3]
    mu = s.mu
    s.on_frame_done(full_job_wait_ms=10.0)                    # > theta = 5.6 ms
    assert abs(s.mu - mu * 0.97) < 1e-9
    s.on_frame_done(full_job_wait_ms=0.0)
    assert abs(s.mu - (mu * 0.97 + 0.01)) < 1e-9
    s.on_frame_skipped()
    assert abs(s.mu - (mu * 0.97 + 0.01) * 0.9) < 1e-9


def test_merge_rule():
    W, H, s_full = 1916, 1078, 0.434
    left_top = (0, 0, 986, 569)
    small = (100, 100, 20, 20, 0.6, 2)                         # 8.7 px in the network input
    large = (100, 100, 120, 80, 0.9, 2)                        # 42.6 px: left to the full frame
    cut = (970, 100, 16, 16, 0.7, 2)                           # touches the inner right border
    assert keep_tile_detection(small, left_top, W, H, s_full)
    assert not keep_tile_detection(large, left_top, W, H, s_full)
    assert not keep_tile_detection(cut, left_top, W, H, s_full)
    out = merge_frame([(101, 101, 20, 20, 0.5, 2)], [(left_top, [small, large])], W, H, s_full)
    assert len(out) == 1 and out[0][4] == 0.6                 # duplicate removed by NMS, best kept
