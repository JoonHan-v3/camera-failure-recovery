"""Parallel worker: info pkls for a slice of {pilot train scenes (every 70th by name)} U {all val scenes}.
usage: build_infos_par.py <worker> <n_workers>"""
import sys, os, json
sys.path.insert(0, '.')
w, n = int(sys.argv[1]), int(sys.argv[2])
from nuscenes.utils import splits
pilot = sorted(splits.train)[::70][:10]
names = sorted(pilot + list(splits.val))
mine = set(names[w::n])
splits.train = [s for s in pilot if s in mine]
splits.val = [s for s in splits.val if s in mine]
import mmcv
_track = mmcv.track_iter_progress
def track(it, *a, **k):          # converter loop only: keep samples of this worker's scenes
    it = list(it)
    if it and isinstance(it[0], dict) and 'scene_token' in it[0] and 'data' in it[0]:
        keep = {sc['token'] for sc in NUSC[0].scene if sc['name'] in mine}
        it = [x for x in it if x['scene_token'] in keep]
    return _track(it, *a, **k)
mmcv.track_iter_progress = track
import nuscenes.nuscenes as nn
NUSC = []
_init = nn.NuScenes.__init__
def init(self, *a, **k):
    _init(self, *a, **k); NUSC.append(self)
nn.NuScenes.__init__ = init
from data_converter import nuscenes_converter as nc
root = os.path.expanduser('~/storage/nuscenes_trainval_root')
nc.create_nuscenes_infos(root, f'w{w}', version='v1.0-trainval', max_sweeps=10)
print('INFOS_DONE')
