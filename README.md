# Partial Registration Network
Fork of PRNet 

## pip list

```
> pip list
Package            Version                
------------------ -----------------------
attrs              19.3.0                 
backcall           0.1.0                  
bleach             3.1.4                  
decorator          4.4.2                  
defusedxml         0.6.0                  
entrypoints        0.3                    
h5py               2.10.0                 
importlib-metadata 1.6.0                  
ipykernel          5.2.0                  
ipython            7.13.0                 
ipython-genutils   0.2.0                  
ipywidgets         7.5.1                  
jedi               0.16.0                 
Jinja2             2.11.1                 
joblib             0.14.1                 
jsonschema         3.2.0                  
jupyter-client     6.1.2                  
jupyter-core       4.6.3                  
MarkupSafe         1.1.1                  
mistune            0.8.4                  
nbconvert          5.6.1                  
nbformat           5.0.5                  
notebook           6.0.3                  
numpy              1.18.2                 
open3d             0.9.0.0                
pandocfilters      1.4.2                  
parso              0.6.2                  
pexpect            4.8.0                  
pickleshare        0.7.5                  
Pillow             7.0.0                  
pip                20.0.2                 
pkg-resources      0.0.0                  
prometheus-client  0.7.1                  
prompt-toolkit     3.0.5                  
ptyprocess         0.6.0                  
Pygments           2.6.1                  
pyrsistent         0.16.0                 
python-dateutil    2.8.1                  
pyzmq              19.0.0                 
scikit-learn       0.22.2.post1           
scipy              1.4.1                  
Send2Trash         1.5.0                  
setuptools         39.0.1                 
six                1.14.0                 
sklearn            0.0                    
terminado          0.8.3                  
testpath           0.4.4                  
torch              1.5.0                  
torchvision        0.6.0.dev20200327+cu101
tornado            6.0.4                  
tqdm               4.43.0                 
traitlets          4.3.3                  
wcwidth            0.1.9                  
webencodings       0.5.1                  
widgetsnbextension 3.5.1                  
zipp               3.1.0 
```


## 

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name "demo" --svd_on_gpu --model_path checkpoints/demo/models/model.74.t7

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_NeurIPS,
	  title={PRNet: Self-Supervised Learning for Partial-to-Partial Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {33rd Conference on Neural Information Processing Systems (To appear)},
	  year={2019}
	}


## License
MIT License
