"""SparseBEV glue: the shared CorruptFrontCamera step (code identical to Member 2's streampetr_adapter.py, which Member 1 also
reuses) plus the mmdet pipeline transform that inserts it right after LoadMultiViewImageFromFiles.

VERIFIED on SparseBEV d341052 / mmdet3d v1.0.0rc6: LoadMultiViewImageFromFiles(to_float32=False) yields results['img'] = six HxWx3
uint8 BGR arrays in converter camera order (CAM_FRONT first). Upstream get_data_info supplies img_timestamp as FLOAT SECONDS
(sample_data.timestamp / 1e6) and the model uses it (time_diff in sparsebev_transformer.py), so it is left untouched and the
adapter runs with ts_mode='abs_s' (round(t*1e6) is exact for these values; the runner asserts it against the integer-us info).
The outage is applied BEFORE RandomTransformImage (resize/crop), the GPU normalization and the backbone.
In single-GPU test mode the loader (load_online) decodes only the six CURRENT images; history comes from the model's
filename-keyed feature cache (see run_sparsebev.py), so a corrupted frame stays corrupted as history by construction."""
from mmdet.datasets import PIPELINES
from .streampetr_adapter import CorruptFrontCamera, cam_from_path  # noqa: F401  (shared, byte-identical)


@PIPELINES.register_module()
class OutageFrontCamera:
    STEP = None

    def __call__(self, results):
        assert OutageFrontCamera.STEP is not None, "runner must set OutageFrontCamera.STEP"
        return OutageFrontCamera.STEP(results)
