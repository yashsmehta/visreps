"""NSD-synthetic with in-dataset 50/50 layer selection: clip32 vs default1000 x BN source."""
import sys, json, pandas as pd
from dotenv import load_dotenv
load_dotenv()
import visreps.utils as utils, visreps.evals as evals

frames = []
for label, ckpt, cfg_id in [("clip32", "/data/ymehta3/clip_pca", 32),
                            ("default1000", "/data/ymehta3/default", 1000)]:
    for bn in ("checkpoint", "nsd"):
        cfg = utils.validate_config(utils.load_config("configs/eval/base.json", [
            "mode=eval", "neural_dataset=nsd_synthetic", "analysis=rsa", "layer_source=split",
            f"bn_calibration_source={bn}", f"checkpoint_dir={ckpt}", f"cfg_id={cfg_id}", "seed=1",
            "eval_checkpoint_at_epoch=20", "log_expdata=false", "bootstrap=false",
            "verbose=false", "batchsize=128", "num_workers=8"]))
        df = evals.eval(cfg)
        df["model"], df["bn"] = label, bn
        df["layer_selection_scores"] = df["layer_selection_scores"].apply(json.dumps)
        frames.append(df)
        print(f"DONE {label} bn={bn}", flush=True)
pd.concat(frames, ignore_index=True).drop(columns=["bootstrap_scores"], errors="ignore").to_csv(sys.argv[1], index=False)
