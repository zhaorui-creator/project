import os
source_path='/home/dl/Downloads/indoorCVPR_09/images/'
target_path='/home/dl/Downloads/indoorCVPR_09/images1/'

i=0
for img in sorted(os.listdir(source_path)):
    i=i+1
    name=os.path.splitext(img)
    img_segment=name[0]
    org_name=os.path.join(source_path,img)
    
    changed_name=target_path+str(i)+'.jpg'
    
        
    os.rename(org_name,changed_name)