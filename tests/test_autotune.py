import json

import numpy as np

import tinyfuse as tf
from tinyfuse.autotune import TuningDB, candidate_tiles, main, matmul_shapes, tune_matmul
from tinyfuse.codegen import CodegenOptions, MatmulTile
from tinyfuse.models import mlp2, mlp2_shapes, np_mlp2, random_inputs
from tinyfuse.runtime import native_vector_width
from conftest import assert_close

VW = native_vector_width()


def test_candidates_respect_register_budget_and_problem_size():
    cs = candidate_tiles(10, 40, 7, VW, max_candidates=500)
    regs = 32 if VW >= 16 else 16
    assert len(cs) == len(set(cs))
    for t in cs:
        assert t.mr * t.nv + t.nv + 1 <= regs
        assert t.mb % t.mr == 0 and t.nb % (t.nv * VW) == 0 and t.kb <= 7


def test_tune_matmul_picks_fastest_correct_tile():
    tile, gf, res = tune_matmul(1, 20, 33, 17, max_candidates=4, reps=2, min_time=0.0,
                                 confirm_time=0.0)
    assert len(res) == 4 and gf > 0
    assert "confirm_gflops" in res[0]   # the default is always a finalist
    assert isinstance(tile, MatmulTile)


def test_db_roundtrip_and_compile_uses_tiles(tmp_path):
    path = tmp_path / "t.json"
    sh = mlp2_shapes(8, 16, 24, 4)
    g = tf.trace(mlp2, **sh)
    db = TuningDB(path).tune_graph(g, max_candidates=3, reps=1, min_time=0.0, confirm_time=0.0)
    keys = matmul_shapes(tf.optimize(g))
    assert keys == [(1, 8, 24, 16, "t0=add(acc,in0); t1=relu(t0)"),
                    (1, 8, 4, 24, "t0=add(acc,in0)")]
    data = json.loads(path.read_text())
    assert len(data) == 2
    db2 = TuningDB(path)
    t = db2.get(keys[0], VW)
    assert t is not None
    f = tf.compile(mlp2, tuning=db2, **sh)
    from tinyfuse.codegen.c_backend import clamp_tile
    c = clamp_tile(t, 8, 24, 16, VW)
    assert f"tile mb={c.mb} nb={c.nb} kb={c.kb} mr={c.mr} nr={c.nv * VW}" in f.source
    xs = random_inputs(sh)
    assert_close(f(*xs), np_mlp2(*[x.astype(np.float64) for x in xs]))
    # explicit per-shape tile overrides the database; caller's options not mutated
    forced = MatmulTile(6, VW, 4, 3, 1)
    opts = CodegenOptions(vector_width=VW, tiles={keys[0][:4]: forced})
    f2 = tf.compile(mlp2, tuning=db2, options=opts, **sh)
    assert f"mb=6 nb={VW} kb=4 mr=3" in f2.source
    assert list(opts.tiles) == [keys[0][:4]]


def test_cli(tmp_path, capsys):
    p = tmp_path / "db.json"
    main(["--shape", "8x16x24", "--db", str(p), "--candidates", "2"])
    assert "best" in capsys.readouterr().out
    assert json.loads(p.read_text())


def test_thread_count_is_part_of_the_key(tmp_path):
    db = TuningDB(tmp_path / "t.json")
    shape = (1, 32, 64, 48, "")
    one, four = MatmulTile(6, 16, 16, 6, 1), MatmulTile(12, 32, 16, 6, 1)
    db.record(shape, VW, one, 10.0, 9.0, 3)
    db.record(shape, VW, four, 30.0, 20.0, 3, threads=4)
    assert db.get(shape, VW) == one and db.get(shape, VW, 4) == four
    assert db.get(shape, VW, 2) is None
    assert "threads" not in TuningDB.key(shape, VW)          # old keys stay valid
    g = tf.trace(lambda a, b: a @ b, (32, 48), (48, 64))
    serial = db.tiles_for(g, CodegenOptions(vector_width=VW))
    par = db.tiles_for(g, CodegenOptions(vector_width=VW, parallel=True, max_threads=4))
    assert list(serial.values()) == [one] and list(par.values()) == [four]


def test_parallel_tuning_runs(tmp_path):
    tile, gf, res = tune_matmul(1, 24, 40, 16, max_candidates=3, reps=1, min_time=0.0,
                                confirm_time=0.0, threads=2)
    assert gf > 0 and len(res) == 3
