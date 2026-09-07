"""Compare the completed 2010 run with torchvision pretrained AlexNet."""
import gc
import json
from pathlib import Path
from dotenv import load_dotenv
import torch
from visreps.utils import load_config, validate_config
from visreps.evals import eval


def main():
    load_dotenv()
    torch.set_num_threads(8)
    out = Path('experiments/alexnet_dataset_comparison/results')
    out.mkdir(exist_ok=True)
    for dataset in ['things-behavior', 'nsd']:
        for model in ['imagenet2010_epoch20', 'torchvision_imagenet2012']:
            target = out / f'{model}_{dataset}.json'
            if target.exists():
                continue
            overrides = [f'neural_dataset={dataset}', 'num_workers=8', 'batchsize=128',
                         'log_expdata=false', 'bootstrap=false', 'reconstruct_from_pcs=false']
            if model == 'imagenet2010_epoch20':
                overrides += ['load_model_from=checkpoint', 'cfg_id=1000',
                              'checkpoint_dir=model_checkpoints/alexnet_imagenet2010_cnn_recipe',
                              'checkpoint_model=checkpoint_epoch_20.pth']
            else:
                overrides += ['load_model_from=torchvision', 'model_name=AlexNet',
                              'pretrained_dataset=imagenet1k']
            cfg = validate_config(load_config('configs/eval/base.json', overrides))
            print(f'\nSTART {model} {dataset}', flush=True)
            result = eval(cfg)
            target.write_text(result.to_json(orient='records', indent=2))
            (out / f'{model}_{dataset}_config.json').write_text(json.dumps(overrides, indent=2))
            print(f'SAVED {target}', flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
