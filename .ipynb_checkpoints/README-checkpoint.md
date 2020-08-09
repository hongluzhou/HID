# HID: Hierarchical Multiscale Representation Learning for Information Diffusion  
         
If this code helps with your work, please cite:      
                
Honglu Zhou, Shuyuan Xu, Zuohui Fu, Gerard de Melo, Yongfeng Zhang and Mubbasir Kapadia. [HID: Hierarchical Multiscale Representation Learning for Information Diffusion](https://www.ijcai.org/Proceedings/2020/0468.pdf). In International Joint Conference on Artificial Intelligence (IJCAI), 2020.      
      
```
@inproceedings{zhou2020hid,
    title = {{HID: Hierarchical Multiscale Representation Learning for Information Diffusion}},
    author = {Zhou, Honglu and Xu, Shuyuan and Fu, Zuohui and de Melo, Gerard and Zhang, Yongfeng and Kapadia, Mubbasir},
    booktitle = {IJCAI},
    year = 2020
}
```
      
## Dataset 
Please go to data folder to find the download links of datasets used in the paper. 
      
      
## Code 
* hid.py is the framework.  
* utils.py has many general-purpose function including "upscaling" and "downscaling" (both are called by hid.py). 
* metrics.py has function implementation for various metrics.   
* test.py is the code to run testing and get the testing performance, by calling "test" function from specific model (e.g. CDK).  
   
   
Run 'conda env create -f environment.yml' to create a conda environment that satisfies the package requirement. Check or modify the conda environment name in the first line of 'environment.yml'.   
         
               
## How to run     
Hyper-parameters:  
- s: num_scales   
- p: coarse_portion  
- upscaling operator: upoperator
   
How to run HID without using upscaling and downscaling (just run baseline) for training and learning:    
```time python hid.py --corpus_path=./data/digg_500user/ --output_path=./data/digg_CDK_s0 --num_scales=0 --max_epochs=8000 --diffuser=CDK```
    
How to run HID with upscaling and downscaling for training and learning (e.g. s=2, p=1.2):         
```time python hid.py --corpus_path=./data/digg_500user/ --output_path=./data/digg_CDK_HAC_s2_p1dot2 --num_scales=2 --coarse_portion=1.2 --max_epochs=2666 --diffuser=CDK --upoperator=HAC```    
    
How to check testing performance (first, modify config in test.py):
```time python test.py```
   
   
