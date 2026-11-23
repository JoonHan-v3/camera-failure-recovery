"""mmdet pipeline transform: black CAM_FRONT inside the outage window, inserted directly after
LoadMultiViewImageFromFiles (raw decoded pixels, before NormalizeMultiviewImage / x0.5 resize / backbone).
The runner sets OutageFrontCamera.STEP (a CorruptFrontCamera) before iterating the dataset."""
from mmdet.datasets import PIPELINES
from .adapter import CorruptFrontCamera


@PIPELINES.register_module()
class OutageFrontCamera:
    STEP = None

    def __call__(self, results):
        assert OutageFrontCamera.STEP is not None, "runner must set OutageFrontCamera.STEP"
        return OutageFrontCamera.STEP(results)
