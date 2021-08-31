# Toshiba Research Europe Ltd reserves all rights, as outlined in license.txt
import numpy as np
import cv2
import os

def load_normal_map(dirpath,height,width,scale): 
    if os.path.isdir(dirpath):    
        normal_path = dirpath + '/normals.png'
    else:
        normal_path=dirpath
    nml = cv2.imread(normal_path,-1)
    if nml is None:
        return None
    if scale != 1.0:
        nml = cv2.resize(nml, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  
    assert(nml.shape[0]==height)
    assert(nml.shape[1]==width)
    if nml.dtype=='uint16':
        nml=np.float32(nml)/65535.0
    else:
        nml=np.float32(nml)/255.0
    #opencv bgr crap
    nx=2*nml[:,:,2]-1
    #Flipping to our axis
    ny=-(2*nml[:,:,1]-1)
    nz=-(2*nml[:,:,0]-1)

    nmag=np.sqrt(nx*nx+ny*ny+nz*nz)
    nmag[nmag==0]=1
    nml=np.stack((nx/nmag,ny/nmag,nz/nmag), axis=2)            
    #print("Normal Image shape is ", nml.shape)
    return nml
def out_normal_image(filename, normap):
    assert(len(normap.shape)==3)
    assert(normap.shape[2]==3)
    MM=65535
    normcpy= normap.copy()
    normcpy[:,:,1:3]*=-1
    normcpy=MM*(normcpy+1)/2   
    normcpy[normcpy>MM]=MM
    normcpy[normcpy<0]=MM
    normcpy=cv2.cvtColor(normcpy,cv2.COLOR_BGR2RGB)    
    cv2.imwrite(filename, normcpy.astype(np.uint16))
    
def out_normal_map(filename, N,maskindx,height,width):   
    normap = (1e6)*np.ones((height*width,3), np.float32) 
    for kk in range(3):
        normap[maskindx,kk]=N[:,kk]       
   
    normap=np.reshape(normap,(height,width,3))     
    out_normal_image(filename, normap)
    
def read_gt_depth(inputDir, scale=1.0):    
    filepath=inputDir+'zgt.npz'      
    if  not os.path.isfile(filepath):       
            return None    
    npzfile = np.load(filepath)
    Z=npzfile['zgt']
    zgt_int=npzfile['zgt_int']
   
    if scale != 1.0:
        Z = cv2.resize(Z, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  
        zgt_int = cv2.resize(zgt_int, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return Z, zgt_int

def export_ply_points(name, x,y,z,triangle_list):       
    nverts=x.shape[0]
    assert(len(x.shape)==1)
    #print('x.shape ',x.shape)
    
    ntriangles=triangle_list.shape[0]
    one_v=np.ones((nverts))
    #print('nverts ',nverts)
    print('exporting ply in [%s], NV %d NF %d'%(name,nverts,ntriangles))    
   
    v = np.vstack((x,y,z))
    #print('v.shape ',v.shape)
    v=v.transpose()   
    #print(v)

    fid= open(name,"w+")
    fid.write("ply\nformat ascii 1.0\nelement vertex %d \nproperty float x\nproperty float y\nproperty float z\n" %(nverts))# 
    fid.write("element face %d\nproperty list uint8 int32 vertex_indices\nend_header\n" %(ntriangles))     
    
    np.savetxt(fid,v,fmt='%f %f %f')
    np.savetxt(fid,triangle_list,fmt='%d %d %d %d')
    fid.close()
def getDistorioNmatrices(filecalib, width,height,scale):
    cv_file = cv2.FileStorage(filecalib, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()    
    K[0:2,0:3]*=scale
    #print('K',K)
    #exit(0)
    #print('D',D)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (width,height), 5)
    return mapx, mapy 
def readNFSetup(inputDir, scale):
    # read light positions and intensity
    if os.path.isdir(inputDir):
        setupfile=inputDir+'led_params.txt'
        if  not os.path.isfile(setupfile):
            setupfile=inputDir+'setup.txt'  
    else:
        setupfile=inputDir
    with open(setupfile) as f:
        data = f.read()
    lines = data.rstrip().split('\n') # rstrip() removes any trailing '\n', which would result in an extra empty line
    # 1st line: header, 2nd: image info, from 3rd: light info
    tokens = lines[1].split(' ')
    N_img, nrows, ncols, f, x0, y0, mean_distance = tokens
    
    N_img=int(N_img)
    nrows=int(nrows)
    ncols=int(ncols)
    f=float(f)
    x0=float(x0)
    y0=float(y0)
    
    lightsInfo = lines[2:]
    numLights = len(lightsInfo)
    assert(numLights==int(N_img))
    Lpos = np.zeros ((numLights,3), np.float32)
    Ldir = np.zeros ((numLights,3), np.float32)
    Phi = np.zeros((numLights,3), np.float32)
    mu = np.zeros((numLights,1), np.float32)
    mean_distance=float(mean_distance) # in mm
    
    for i,l in enumerate(lightsInfo):
        tokens = l.split(' ')          
        Lpos[i,:] = np.array(tokens[0:3], dtype=np.float32)   
        Ldir[i,:] = np.array(tokens[3:6], dtype=np.float32)             
        #old-single phi value-tokens are 9+ empty string at the end but not at last row
        if (len(tokens)==10) or (len(tokens)==9):
            Phi[i,:] = float(tokens[6])
            mu_=float(tokens[7])
        else:
            Phi[i,:] = np.array(tokens[6:9], dtype=np.float32)
            mu_=float(tokens[9]) 
        mu[i,0]=mu_
            #print('mu is ',mu)
    f=float(f)   
    ncols = int(ncols*scale)
    nrows = int(nrows*scale)
    x0 *= scale
    y0 *= scale    
    f *= scale
    
    return N_img, ncols, nrows, f, x0, y0, mean_distance, Lpos, Ldir, Phi, mu
def writeNfSetup(filename, N_img, ncols, nrows, f, x0, y0, mean_distance, Sfull):
    fid = open(filename, "w")
    #header
    fid.write("N_img nrows ncols f x0 y0 mean_distance sfull:[Lpos Ldir Phi mu 0]\n")
    str_header='%d %d %d %f %f %f %f\n'%(N_img, ncols, nrows, f, x0, y0, mean_distance)
    fid.write(str_header)
    np.savetxt(fid, Sfull,fmt='%.4f',)
    fid.close()
def load_images(dirpath,height, width,numLight):
# read images     
    I = np.zeros((height, width,3,numLight), np.float32)    
   
    for i in range(numLight):         
        image_path = os.path.join(dirpath ,  '%02d.png' % (i + 1))           
        if not os.path.isfile(image_path): 
            print('could not find images in ',dirpath)
            exit(0)
        cv2_im = cv2.imread(image_path, -1)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            
        if cv2_im.dtype=='uint16':
            cv2_im=np.float32(cv2_im)/65535.0
        else:
            cv2_im=np.float32(cv2_im)/255.0   
              
        cv2_im=cv2.resize(cv2_im, (width,height), interpolation=cv2.INTER_NEAREST)   
        I[:,:,:,i] = cv2_im
       
    return I
def load_images_raw(dirpath,height, width,numLight):
# read images     
    I = np.zeros((height, width, numLight), np.float32)    
   
    for i in range(numLight):        
        image_path = dirpath + '/' + '%02d.png' % (i + 1)
        if not os.path.isfile(image_path):
            image_path = dirpath + '/' + '%05d.png' % (i)
        if not os.path.isfile(image_path):
            image_path = dirpath + '/' + '%05d.tif' % (i)
        if not os.path.isfile(image_path):    
            image_path = dirpath + '/' +'led_prev_%02d.png'%i
        if not os.path.isfile(image_path): 
            print('could not find images in ',dirpath)
            exit(0)
        cv2_im = cv2.imread(image_path, -1)
        #print(cv2_im.shape)
        assert(len(cv2_im.shape)==2)
        #cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)            
        if cv2_im.dtype=='uint16':
            cv2_im=np.float32(cv2_im)/65535.0
        else:
            cv2_im=np.float32(cv2_im)/255.0   
            #assert size   
            #assert(cv2_im.shape[0]==height)
            #assert(cv2_im.shape[1]==width)     
        cv2_im=cv2.resize(cv2_im, (width,height), interpolation=cv2.INTER_NEAREST)   
            #cv2.imshow('a',cv2_im)
            #cv2.waitKey(0)
        I[:,:,i] = cv2_im       
    return I
def load_mask_indx(dirpath, ignoreBoundaries=False,scale=1):
    if os.path.isdir(dirpath):
        mask_path=os.path.join(dirpath , 'mask.png' )        
    else:
        mask_path=dirpath
    mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)  
    if scale != 1.0:
        mask_img = cv2.resize(mask_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) 
    mask_img[mask_img>0]=1
    nShape = np.shape(mask_img) 
    height= nShape[0]  
    width = nShape[1]     

    Nmask_old=np.sum(np.sum(mask_img))
    #erode a bit to avoid hard boundaries 
    if ignoreBoundaries:  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        mask_old=np.copy(mask_img)
        mask_img = cv2.erode(mask_old, kernel) 
        #mask=mask_old-mask
        Nmask_new=np.sum(np.sum(mask_img))
        print('mask eroded from %d to %d'%(Nmask_old,Nmask_new))
    
    #masks=[]
    #mask_tags=[]

    validsub = np.where(mask_img>0)
    validind = validsub[0]*width + validsub[1]    
  
    return nShape,mask_img, validind    
def save_emap_jet(filename,Emap,thress_red):
    #=30.0  
    background_mask=Emap<0      
    #Emap*=(90.0/thress_red)
    Emap*=(255.0/thress_red)
    Emap[Emap>255.0]=255.0
    Emap=np.uint8(Emap) 
    Emap = cv2.applyColorMap(Emap, cv2.COLORMAP_JET)
    Emap[background_mask]=255
    cv2.imwrite(filename,Emap)  


if __name__ == '__main__':
    input_dir = r'/data/real_data/Luces/data/Ball/led_params.txt'
    readNFSetup(input_dir, 1)
