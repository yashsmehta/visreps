"""Compare every PNG and loader tensor with the original HDF5/reference recipe."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from visreps.dataloaders.neural import _make_loader
from visreps.dataloaders.obj_cls import get_transform
from experiments.nsd_synthetic_poc.preprocessing import SyntheticReferenceTransform

OUT=Path('experiments/nsd_synthetic_poc/preprocessing_audit')
DATA=Path('/data/shared/datasets/gifford2025.nsd_synthetic')


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    names=pd.read_csv(DATA/'nsddata/experiments/nsdsynthetic/nsdsyntheticimageinformation.csv')['Image'].tolist()[:220]
    pngs={sid:str(Path('datasets/neural/nsd_synthetic/stimuli')/(sid+'.png')) for sid in names}
    reference=SyntheticReferenceTransform()
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    expected={}; blanks_old=[]; blanks_ref=[]; rows=[]
    montage_names=['natscene_1','spiral_A_sf1_1','word4_pos1_1','word4_pos2_1','word4_pos3_1','word4_pos5_1']
    canvas=Image.new('RGB',(1000,len(montage_names)*180),(255,255,255)); draw=ImageDraw.Draw(canvas)
    with h5py.File(DATA/'nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5') as f:
        for i,sid in enumerate(names):
            raw=f['imgBrick'][i]
            png=Image.open(pngs[sid]).convert('RGB')
            np.testing.assert_array_equal(raw,np.asarray(png))
            assert raw.shape==(714,1360,3) and raw.dtype==np.uint8
            # Independent explicit PIL crop/resize and tensor arithmetic oracle.
            gamma=(np.sqrt(raw/255)*255).astype(np.uint8)
            cropped=Image.fromarray(gamma).crop((323,0,1037,714)).resize((224,224),Image.Resampling.BILINEAR)
            tensor=torch.from_numpy(np.asarray(cropped).copy()).permute(2,0,1).float()/255
            expected[sid]=(tensor-mean)/std
            actual=reference(png)
            torch.testing.assert_close(actual,expected[sid],rtol=0,atol=0)
            assert actual.shape==(3,224,224) and actual.dtype==torch.float32 and torch.isfinite(actual).all()
            old=get_transform()(png)
            old_blank=bool((old-old[:,0,0,None,None]).abs().max()==0)
            ref_blank=bool((actual-actual[:,0,0,None,None]).abs().max()==0)
            if old_blank: blanks_old.append(sid)
            if ref_blank: blanks_ref.append(sid)
            rows.append(dict(stimulus=sid,legacy_spatially_constant=old_blank,reference_spatially_constant=ref_blank))
            if sid in montage_names:
                row=montage_names.index(sid); y=row*180
                draw.text((5,y+4),sid,fill='black')
                shown=png.copy(); shown.thumbnail((450,140)); canvas.paste(shown,(5,y+28))
                old_img=Image.fromarray(((old*std+mean).clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8))
                old_img=old_img.resize((140,140)); canvas.paste(old_img,(480,y+28))
                new_img=cropped.resize((140,140)); canvas.paste(new_img,(720,y+28))
                draw.text((480,y+4),'Previous input',fill='black');draw.text((720,y+4),'Reference input',fill='black')
    loader=_make_loader(pngs,reference,batch=32,workers=0)
    seen=[]
    for tensors,ids in loader:
        torch.testing.assert_close(tensors,torch.stack([expected[sid] for sid in ids]),rtol=0,atol=0)
        seen.extend(ids)
    assert seen==sorted(names)
    torch.save(dict(ids=seen,tensors=torch.stack([expected[sid] for sid in seen])),OUT/'expected_model_inputs.pt')
    canvas.save(OUT/'input_comparison.png')
    pd.DataFrame(rows).to_csv(OUT/'stimulus_checks.csv',index=False)
    report=dict(pngs_exactly_match_hdf5=220,loader_tensors_exactly_match_reference=220,
                raw_shape=[714,1360,3],model_input_shape=[3,224,224],
                legacy_blank_stimuli=blanks_old,reference_blank_stimuli=blanks_ref,
                reference_source='https://github.com/gifale95/NSD-synthetic/blob/0628d49f5441e5429fcc26468d0938d368e3ed45/paper_figure_4/01_extract_nsdsynthetic_image_features.py',
                transform=repr(reference))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
