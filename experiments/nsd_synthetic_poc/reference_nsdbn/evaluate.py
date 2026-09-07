"""NSD-synthetic, reference transform, NSD-recalibrated BN: clip32 vs default1000, seed 1."""
import sys, pandas as pd
from dotenv import load_dotenv
load_dotenv()
import visreps.utils as utils, visreps.evals as evals

frames = []
for label, ckpt, cfg_id in [("clip32", "/data/ymehta3/clip_pca", 32),
                            ("default1000", "/data/ymehta3/default", 1000)]:
    overrides = ["mode=eval", "neural_dataset=nsd_synthetic", "analysis=rsa",
                 "bn_calibration_source=nsd", f"checkpoint_dir={ckpt}", f"cfg_id={cfg_id}",
                 "seed=1", "eval_checkpoint_at_epoch=20", "log_expdata=false",
                 "bootstrap=false", "verbose=false", "batchsize=128", "num_workers=8"]
    cfg = utils.validate_config(utils.load_config("configs/eval/base.json", overrides))
    df = evals.eval(cfg)
    df["model"] = label
    frames.append(df)
    print(f"DONE {label}", flush=True)

out = pd.concat(frames, ignore_index=True)
out.drop(columns=["layer_selection_scores", "bootstrap_scores"], errors="ignore").to_csv(sys.argv[1], index=False)
print("saved", sys.argv[1])
