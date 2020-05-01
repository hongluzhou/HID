# HID
   
### Note: The code will be released after formal publication.   
      
* hid.py is the framework.  
* utils.py has many general-purpose function including "upscaling" and "downscaling" (both are called by HID.py). 
* metrics.py has function implementation for various metrics.   
* test.py is the code to run testing and get the testing performance, by calling "test" function from specific model (e.g. CSDK).     
   
   
Run 'conda env create -f environment.yml' to create a conda environment that satisfies the package requirement. Check or modify the conda environment line in the first line of 'environment.yml'.   
   
## How to run   
How to run HID without using upscaling and downscaling (just run baseline) for training and learning:     

	time python hid.py --corpus_path=./data/digg/ --output_path=./data/digg_CDK_0scale --num_scales=0 --max_epochs=8000 --diffuser=CDK 
	
How to run HID with upscaling and downscaling for training and learning:     

	time python hid.py --corpus_path=./data/digg/ --output_path=./data/digg_CDK_0scale_HAC_s1_p2 --num_scales=1 --coarse_portion=2 --max_epochs=4000 --diffuser=CDK --upoperator=HAC
  
How to check testing performance (first, modify config in test.py):

	time python test.py
   
   
