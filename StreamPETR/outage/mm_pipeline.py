"""mmdet3d pipeline transform: black CAM_FRONT inside the outage window, inserted directly after
LoadMultiViewImageFromFiles (raw decoded pixels, before resize/normalize/feature extraction).
The runner sets OutageFrontCamera.STEP (a CorruptFrontCamera) before building the dataloader."""
from mmdet.datasets import PIPELINES
from .streampetr_adapter import CorruptFrontCamera


@PIPELINES.register_module()
class OutageFrontCamera:
    STEP = None

    def __call__(self, results):
        assert OutageFrontCamera.STEP is not None, "runner must set OutageFrontCamera.STEP"
        return OutageFrontCamera.STEP(results)
