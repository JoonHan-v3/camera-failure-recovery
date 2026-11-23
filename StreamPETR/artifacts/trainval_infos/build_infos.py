"""Info pkls (StreamPETR converter) for the frozen pilot (10 train scenes, every 70th by name) + all 150 val scenes."""
import sys, os, json
sys.path.insert(0, '.')
from nuscenes.utils import splits
pilot = sorted(splits.train)[::70][:10]
json.dump(pilot, open('/home/rockhan/Documents/UM/rob535/projects/StreamPETR/artifacts/trainval_infos/pilot_scenes.json', 'w'))
splits.train = pilot                      # converter reads splits.train / splits.val
from data_converter import nuscenes_converter as nc
root = os.path.expanduser('~/storage/nuscenes_trainval_root')
nc.create_nuscenes_infos(root, 'nuscenes2d_pilotval', version='v1.0-trainval', max_sweeps=10)
print('INFOS_DONE')
