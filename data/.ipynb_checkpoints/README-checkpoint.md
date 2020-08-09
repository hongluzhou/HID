# Download    
* memetracker: https://drive.google.com/file/d/1KNrBirsSjvPECXIZaoPJRXwVIWgpssPq/view?usp=sharing    
* twitter: https://drive.google.com/file/d/1CeUMNt3p7nDclSHj-Qet7nPIGm2-3OUn/view?usp=sharing
* digg: https://drive.google.com/file/d/1Sft1hzT6lqMpqGfb19dIHO-aHDsXrdTT/view?usp=sharing
    
# Data
Training data (number after 'D' represents the scale index. E.g, D0 is original finest scale):    
* D0_diffpath_user.pickle    
* D0_diffpath_time.pickle    
* D0_diffpath_info.pickle    
* D0_diffpath_info_reverse.pickle    
* D0_uid_uname.pickle    
* D0_uid_uname_reverse.pickle    
* D0_infoid_infoname.pickle    
* D0_infoid_infoname_reverse.pickle    

corpus = (diffpath_user_D0, diffpath_time_D0,
          diffpath_info_D0, diffpath_info_reverse_D0,
          infoid_infoname_D0, infoid_infoname_reverse_D0,
          uid_uname_D0, uid_uname_reverse_D0)
    
    
Testing data:    
* diffpath_user_test.pickle    
* diffpath_time_test.pickle    
* diffpath_info_test.pickle    
* diffpath_info_reverse_test.pickle    
    
    
# Data Format Details    
* diffpath_user: A list of lists; len(D0_diffpath_user) is the number of diffusion paths. Each inner list is a diffusion path, i.e., the user sequence of this diffusion path. The first user is the source user.    
    
* diffpath_time: A list of lists; len(D0_diffpath_time) is the number of diffusion paths. Each inner list is a diffusion path corresponds to D0_diffpath_user, i.e., the time sequence of this diffusion path.     
    
* diffpath_info: Dictionary; mapping from index of each diffusion path in D0_diffpath_user/D0_diffpath_time to that diffusion path's information. Key is the diffusion path index, with value of the information of diffusion path of that index.    
    
* diffpath_info_reverse: Dictionary; mapping from diffusion path's information to the index of that diffusion path in D0_diffpath_user/D0_diffpath_time. Key is the information of diffusion path of that index, with value the diffusion path index.    
    
* uid_uname: Dictionary; mapping from user integer ID to user name. Key is the user integer ID, with value the user's user name. The integer ID starts from 0 and should be used for mapping to user embedding matrix.      
    
* uid_uname_reverse: Dictionary; mapping from user name to user integer ID. Key is the user's user name, with value the user's integer ID.    
* infoid_infoname: Dictionary; mapping from information integer ID to the corresponding information string. Key is the information integer ID, with value the corresponding information string. The integer ID starts from 0 and should be used for mapping to information embedding matrix (if there is any).    
    
* infoid_infoname_reverse: Dictionary; mapping from information string to information integer ID.  Key is the information string, with value the corresponding information integer ID.      
    