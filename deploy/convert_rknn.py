import json, glob, sys
from rknn.api import RKNN
platform = sys.argv[1] if len(sys.argv) > 1 else "rk3588"
cfg = json.load(open("deploy_config.json"))
imgs = sorted(glob.glob("calib_images/*.jpg"))
open("calib.txt", "w").write("\n".join(imgs))
rk = RKNN(verbose=False)
rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform,
          quantized_algorithm="normal", quantized_method="channel")
assert rk.load_onnx(model="model_fp32.onnx") == 0
assert rk.build(do_quantization=True, dataset="calib.txt") == 0
assert rk.export_rknn("model_int8.rknn") == 0
print("OK -> model_int8.rknn")
rk.release()
