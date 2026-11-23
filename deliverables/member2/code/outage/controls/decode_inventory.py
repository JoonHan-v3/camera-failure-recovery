"""Control: decode every image the val infos reference (cv2, as mmcv does) and save an inventory.
Run: python -m outage.controls.decode_inventory artifacts/trainval_infos/val_infos.pkl artifacts/controls/decode_inventory.json
Records counts by camera, failures, shape set, file-size min/median/max. Pass: 0 failures and every shape == (900,1600,3)."""
import json, os, pickle, sys, collections
import cv2
from multiprocessing import Pool

cv2.setNumThreads(1)


def chk(p):
    im = cv2.imread(p)
    return p, (None if im is None else tuple(im.shape)), os.path.getsize(p)


def main(infos_pkl, out):
    infos = pickle.load(open(infos_pkl, "rb"))["infos"]
    items = [(cam, c["data_path"]) for i in infos for cam, c in i["cams"].items()]
    with Pool(32) as pl:
        res = pl.map(chk, [p for _, p in items], chunksize=64)
    by_cam = collections.Counter(cam for cam, _ in items)
    shapes = collections.Counter(str(r[1]) for r in res)
    sizes = sorted(r[2] for r in res)
    bad = [r[0] for r in res if r[1] != (900, 1600, 3)]
    rep = dict(infos=infos_pkl, n_samples=len(infos), n_images=len(res), per_camera=dict(by_cam), shapes=dict(shapes),
               failures=len(bad), failed_examples=bad[:5], bytes_min=sizes[0], bytes_median=sizes[len(sizes) // 2],
               bytes_max=sizes[-1], passed=not bad, decoder="cv2 " + cv2.__version__)
    json.dump(rep, open(out, "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "failed_examples"})


if __name__ == "__main__":
    main(*sys.argv[1:3])
