import os
import numpy as np 
from PIL import Image
import cv2
import yaml
import torch


def read_pose_file(pose_file):
    fp = open(pose_file,'r')
    
    fstr = next(fp,None)

    pose = np.zeros((3,4)) 
    for i in range(3):
        fstr = next(fp,None)

        felems = fstr.strip().split()
        pose[i][0] = float(felems[0])
        pose[i][1] = float(felems[1])
        pose[i][2] = float(felems[2])
        pose[i][3] = float(felems[3])

    fp.close()

    return pose


def yaml_construct_opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node,deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


def read_calib(calib_path):
    yfp = open(calib_path,"r")
    next(yfp,None)
    yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', yaml_construct_opencv_matrix,yaml.SafeLoader)
    calib = yaml.safe_load(yfp)
    yfp.close()

    return calib


class RoadVidData:
    def __init__(self, opts):
        self.pose_dir = opts.pose_dir
        self.img_dir = opts.img_dir
        self.device = 'cuda'

        self.load_datas()


        calib = read_calib(opts.calib_yaml)

        self.calib_wid = calib["Camera.width"]
        self.calib_hei = calib["Camera.height"]

        self.Kmat = np.eye(3)
        self.Kmat[0,0] = calib["Camera.fx"]
        self.Kmat[1,1] = calib["Camera.fy"]
        self.Kmat[0,2] = calib["Camera.cx"]
        self.Kmat[1,2] = calib["Camera.cy"]

        self.width = opts.width
        self.height = opts.height
        scale_x = float(opts.width)/float(self.calib_wid)
        scale_y = float(opts.height)/float(self.calib_hei)

        self.img_resize = [opts.width,opts.height]
        self.topleft = [0,0]
        self.img_scale = 1.0
        if scale_x <= scale_y:
            self.img_resize[1] = int(self.calib_hei*scale_x)
            self.topleft[1] = int((opts.height - self.img_resize[1])/2)
            self.img_scale = scale_x
        else:
            self.img_resize[0] = int(self.calib_wid*scale_y)
            self.topleft[0] = int((opts.width - self.img_resize[0])/2)
            self.img_scale = scale_y

        self.scaleKmat = np.eye(3)
        self.scaleKmat[0,0] = self.Kmat[0,0]*self.img_scale
        self.scaleKmat[1,1] = self.Kmat[1,1]*self.img_scale
        self.scaleKmat[0,2] = self.Kmat[0,2]*self.img_scale + float(self.topleft[0])
        self.scaleKmat[1,2] = self.Kmat[1,2]*self.img_scale + float(self.topleft[1])


    def __len__(self):
        return len(self.poseList) - 1


    def load_datas(self):

        fp = open(os.path.join(self.pose_dir,'pose.list'),'r')

        self.imgPathList = []
        self.poseList = []

        bFirst = True
        for line in fp:
            lelems = line.strip().split()
            if len(lelems) < 3:
                continue
            
            if bFirst:
                self.imgPathList.append(os.path.join(self.img_dir,"{0}.jpg".format(lelems[1])))
                bFirst = False

            self.imgPathList.append(os.path.join(self.img_dir,"{0}.jpg".format(lelems[0])))

            pose = read_pose_file(os.path.join(self.pose_dir,lelems[2]))
            self.poseList.append(pose)

        fp.close()


    def get_data(self,idx):
        
        if idx < 0 or idx >= len(self.poseList)-1:
            print("Index:{0} is out of range(0-{1})".format(idx,len(self.poseList)-1))
            return None

        ## Read images (resize, crop, adjust Kmat)
        ref_img = cv2.imread(self.imgPathList[idx+1])
        ref_rgb = self.process_image(ref_img)
        # ref_img = cv2.resize(ref_img,(self.img_resize[0],self.img_resize[1]),interpolation=cv2.INTER_LINEAR)
        # ref_bgr = np.zeros((self.height,self.width,3),dtype=np.uint8)
        # ref_bgr[self.topleft[1]:self.topleft[1]+self.img_resize[1],self.topleft[0]:self.topleft[0]+self.img_resize[0],:] = ref_img
        # ref_rgb = cv2.cvtColor(ref_bgr,cv2.COLOR_BGR2RGB)
        ref_rgb = torch.from_numpy(ref_rgb).permute(2,0,1)
        ref_rgb = ref_rgb.to(self.device)

        img_list = [] 
        img0 = cv2.imread(self.imgPathList[idx])
        rgb0 = self.process_image(img0)
        # img0 = cv2.resize(img0,(self.img_resize[0],self.img_resize[1]),interpolation=cv2.INTER_LINEAR)
        # img0_bgr = np.zeros((self.height,self.width,3),dtype=np.uint8)
        # img0_bgr[self.topleft[1]:self.topleft[1]+self.img_resize[1],self.topleft[0]:self.topleft[0]+self.img_resize[0],:] = img0
        # img0_rgb = cv2.cvtColor(img0_bgr,cv2.COLOR_BGR2RGB)
        rgb0 = torch.from_numpy(rgb0).permute(2,0,1)
        rgb0 = rgb0.to(self.device)
        img_list.append(rgb0.unsqueeze(0))

        img1 = cv2.imread(self.imgPathList[idx+2])
        rgb1 = self.process_image(img1)
        # img1 = cv2.resize(img1,(self.img_resize[0],self.img_resize[1]),interpolation=cv2.INTER_LINEAR)
        # img1_bgr = np.zeros((self.height,self.width,3),dtype=np.uint8)
        # img1_bgr[self.topleft[1]:self.topleft[1]+self.img_resize[1],self.topleft[0]:self.topleft[0]+self.img_resize[0],:] = img1
        # img1_rgb = cv2.cvtColor(img1_bgr,cv2.COLOR_BGR2RGB)
        rgb1 = torch.from_numpy(rgb1).permute(2,0,1)
        rgb1 = rgb1.to(self.device)
        img_list.append(rgb1.unsqueeze(0))


        ## Set pose mat (world to camera)
        ref_pose = np.eye(4)
        ref_pose = torch.from_numpy(ref_pose).float().to(self.device)

        pose_list = []
        pose_0r = np.eye(4)
        pose_0r[:3,:] = self.poseList[idx]
        pose_0r = torch.from_numpy(pose_0r.astype(np.float32)).float().to(self.device)
        pose_list.append(pose_0r.unsqueeze(0))

        pose_1r = np.eye(4)
        for i in range(3):
            for j in range(3):
                pose_1r[i][j] = self.poseList[idx][j][i]
                pose_1r[i][3] -= self.poseList[idx][j][i]*self.poseList[idx][j][3]
        pose_1r = torch.from_numpy(pose_1r.astype(np.float32)).float().to(self.device)
        pose_list.append(pose_1r.unsqueeze(0))


        ## Kmat
        K_pool = {}
        for i in range(6):
            K_pool[(self.height//2**i,self.width//2**i)] = self.scaleKmat.copy().astype('float32')
            K_pool[(self.height//2**i,self.width//2**i)][:2,:] /= 2**i
        
        inv_K_pool = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3,:3] = v
            invK = np.linalg.inv(K44).astype(np.float32)
            inv_K_pool[k] = torch.from_numpy(invK).to(self.device).unsqueeze(0)

        return ref_rgb.unsqueeze(0), img_list, ref_pose.unsqueeze(0), pose_list, inv_K_pool
    

    def process_image(self,img):
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = cv2.resize(img,(self.img_resize[0],self.img_resize[1]),interpolation=cv2.INTER_LINEAR)
        
        bgr = np.zeros((self.height,self.width,3),dtype=np.uint8)
        bgr[self.topleft[1]:self.topleft[1]+self.img_resize[1],self.topleft[0]:self.topleft[0]+self.img_resize[0],:] = img
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

        rgb = rgb.astype(np.float32)/255.0
        rgb = (rgb-mean)/std

        return rgb.astype(np.float32)

    def postprocess_depth(self,depth):

        return depth
        
