import os
import time
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from kmeans_pytorch import kmeans

import voc12.dataloader
from misc import torchutils, imutils
import net.resnet50_cam as resnet50_cam
from torch.utils.data import DataLoader


class_id_to_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def print_tensor(x,n=3):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(round(x[i,j],n),end=' ')
        print()

def save_feature(model, save_dir,voc12_root): ####### save feature from resnet50 at 'cluster/cam_feature/'
    if osp.exists(osp.join(save_dir,'tensor_logits.pt')):
        print("[lpcam:save-features] skipped")
        return

    dataset = voc12.dataloader.VOC12ClassificationDataset('voc12/train.txt', voc12_root=voc12_root)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print(len(data_loader))
    tensor_logits = torch.zeros(len(data_loader),20)
    tensor_label = torch.zeros(len(data_loader),20)
    # tensor_cam = torch.zeros(len(data_loader),20,32,32)
    tensor_feature = {}
    name2id = dict()
    with torch.no_grad():
        for i, pack in enumerate(data_loader):
            img = pack['img'].cuda()
            label = pack['label']
            logit,feature = model(img, with_feature=True)
            name2id[pack['name'][0]] = i
            tensor_logits[i] = logit[0].cpu()
            # tensor_cam[i] = cams[0].cpu()
            tensor_feature[i] = feature[0].cpu()
            tensor_label[i] = label[0]
            # print(i)

    os.makedirs(save_dir,exist_ok=True)
    torch.save(tensor_logits, osp.join(save_dir,'tensor_logits.pt'))
    torch.save(tensor_feature, osp.join(save_dir,'tensor_feature.pt'))
    torch.save(tensor_label, osp.join(save_dir,'tensor_label.pt'))
    np.save(osp.join(save_dir,'name2id.npy'), name2id)

def load_feature_select_and_cluster(model, workspace, feature_dir, mask_dir, load_cluster=False, num_cluster=12,select_thres=0.1,class_thres=0.9,context_thres=0.9,context_thres_low=0.05,tol=5, max_feature_samples=12_000, override=False):
    if osp.exists(osp.join(workspace,'class_ceneters'+'.pt')) and not override:
        return print("[lpcam:load-feature-select-and-cluster] skipped")

    tensor_feature = torch.load(osp.join(feature_dir,'tensor_feature.pt'))
    tensor_label = torch.load(osp.join(feature_dir,'tensor_label.pt'))
    name2id = np.load(osp.join(feature_dir,'name2id.npy'), allow_pickle=True).item()
    id2name = {}
    for key in name2id.keys():
        id2name[name2id[key]] = key

    ##### load model for calc similarity
    w = model.classifier.weight.squeeze().detach().cpu()
    
    ####### feature cluster #####
    centers = {}
    context = {}
    for class_id in range(20):
        print()
        print('class id: ', class_id,', class name:',class_id_to_name[class_id])
        cluster_result_dir = osp.join(workspace,'cluster_result')
        os.makedirs(cluster_result_dir, exist_ok=True)
        
        if load_cluster:
            cluster_centers = torch.load(osp.join(cluster_result_dir,'cluster_centers_'+str(class_id)+'.pt'))
            cluster_centers2 = torch.load(osp.join(cluster_result_dir,'cluster_centers2_'+str(class_id)+'.pt'))
            cluster_ids_x = torch.load(osp.join(cluster_result_dir,'cluster_ids_x_'+str(class_id)+'.pt'))
            cluster_ids_x2 = torch.load(osp.join(cluster_result_dir,'cluster_ids_x2_'+str(class_id)+'.pt'))
        else:
            img_selected = torch.nonzero(tensor_label[:,class_id])[:,0].numpy()
            feature_selected = []
            feature_not_selected = []
            for idx in img_selected:
                name = id2name[idx]
                cam = np.load(osp.join(mask_dir, name+'.npy'), allow_pickle=True).item()
                if "high_res" in cam:
                    mask = cam['high_res']
                    valid_cat = cam['keys']
                else:
                    # AffinityNet/Puzzle/OC-CSE format. In this case,
                    # we revert the inclusion of a bg class.
                    mask = cam["hr_cam"]
                    valid_cat = cam['keys'][1:] - 1

                feature_map = tensor_feature[idx].permute(1,2,0)
                size = feature_map.shape[:2]
                mask = F.interpolate(torch.tensor(mask).unsqueeze(0),size)[0]
                for i in range(len(valid_cat)):
                    if valid_cat[i]==class_id:
                        mask = mask[i]
                        position_selected = mask>select_thres
                        position_not_selected = mask<select_thres
                        feature_selected.append(feature_map[position_selected])
                        feature_not_selected.append(feature_map[position_not_selected])
            feature_selected = torch.cat(feature_selected,0)
            feature_not_selected = torch.cat(feature_not_selected,0)

            if feature_selected.shape[0] > max_feature_samples:
                p = torch.randperm(feature_selected.shape[0])[:max_feature_samples]
                feature_selected = feature_selected[p, ...]

            if feature_not_selected.shape[0] > max_feature_samples:
                p = torch.randperm(feature_not_selected.shape[0])[:max_feature_samples]
                feature_not_selected = feature_not_selected[p, ...]

            print(f"features selected    : {feature_selected.shape} {feature_selected.dtype}")
            print(f"features not selected: {feature_not_selected.shape} {feature_not_selected.dtype}")
            
            cluster_ids_x, cluster_centers = kmeans(X=feature_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol, max_iteration=501)
            cluster_ids_x2, cluster_centers2 = kmeans(X=feature_not_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol, max_iteration=501)
        
            torch.save(cluster_centers.cpu(), osp.join(cluster_result_dir,'cluster_centers_'+str(class_id)+'.pt'))
            torch.save(cluster_centers2.cpu(), osp.join(cluster_result_dir,'cluster_centers2_'+str(class_id)+'.pt'))
            torch.save(cluster_ids_x.cpu(), osp.join(cluster_result_dir,'cluster_ids_x_'+str(class_id)+'.pt'))
            torch.save(cluster_ids_x2.cpu(), osp.join(cluster_result_dir,'cluster_ids_x2_'+str(class_id)+'.pt'))
        

        ###### calc similarity
        sim = torch.mm(cluster_centers,w.T)
        prob = F.softmax(sim,dim=1)
        
        ###### select center
        selected_cluster = prob[:,class_id]>class_thres
        cluster_center = cluster_centers[selected_cluster]
        centers[class_id] = cluster_center.cpu()
        
        print(f"-> FG similarity matrix ({class_id_to_name[class_id]})")
        print_tensor("softmax(Centroid @ W.T) =", prob[:, class_id].numpy().round(2))
        for i in range(num_cluster):
           print(selected_cluster[i].item(), round(prob[i,class_id].item(), 3), torch.sum(cluster_ids_x == i).item())
        
        ###### calc similarity
        sim = torch.mm(cluster_centers2,w.T)
        prob = F.softmax(sim,dim=1)
        
        ###### select context
        selected_cluster = (prob[:,class_id]>context_thres_low)*(prob[:,class_id]<context_thres)
        cluster_center2 = cluster_centers2[selected_cluster]
        context[class_id] = cluster_center2.cpu()
        
        ##### print similarity matrix
        print("-> BG similarity matrix (context)")
        print_tensor("softmax(Centroid @ W.T) =", prob[:, class_id].numpy().round(2))
        for i in range(num_cluster):
            print(f"C#{i}", selected_cluster[i].item(), round(prob[i,class_id].item(),3), torch.sum(cluster_ids_x2==i).item())

    # torch.save(centers.cpu(), osp.join(workspace+'class_ceneters'+'.pt'))
    torch.save(centers, osp.join(workspace,'class_ceneters'+'.pt'))
    torch.save(context, osp.join(workspace,'class_context'+'.pt'))

def make_lpcam(model,workspace,lpcam_out_dir,voc12_root,list_name='voc12/train.txt'):
    print("[lpcam:make-lpcam] start")
    cluster_centers = torch.load(osp.join(workspace,'class_ceneters'+'.pt'))
    cluster_context = torch.load(osp.join(workspace,'class_context'+'.pt'))
    
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(list_name, voc12_root=voc12_root,scales=(1.0, 0.5, 1.5, 2.0))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    start_time = time.time()
    with torch.no_grad():
        for i, pack in enumerate(tqdm(data_loader)):
            imgs = pack['img']
            label = pack['label'][0]
            img_name = pack['name'][0]
            size = pack['size']
            valid_cat = torch.nonzero(label)[:, 0].numpy()

            save_path = os.path.join(lpcam_out_dir, img_name.replace('jpg','npy'))
            if osp.exists(save_path):
                continue

            # print(">", img_name)

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            features = []
            for img in imgs:
                _, feature = model(img[0].cuda(), with_feature=True)
                feature = feature.cpu()
                features.append((feature[0]+feature[1].flip(-1)))
                
            
            strided_cams = []
            highres_cams = []
            for class_id in valid_cat:
                # print("class>", class_id)
                strided_cam = []
                highres_cam = []
                for feature in features:
                    h,w = feature.shape[1],feature.shape[2]    
                    cluster_feature = cluster_centers[class_id]
                    att_maps = []
                    for j in range(cluster_feature.shape[0]): 
                        cluster_feature_here = cluster_feature[j].repeat(h,w,1) # .cuda()
                        feature_here = feature.permute((1,2,0)).reshape(h,w,2048)
                        attention_map = F.cosine_similarity(feature_here,cluster_feature_here,2).unsqueeze(0).unsqueeze(0)
                        att_maps.append(attention_map.cpu())
                    att_map = torch.mean(torch.cat(att_maps,0),0,keepdim=True) # .cuda()
                    
                    context_feature = cluster_context[class_id]
                    if context_feature.shape[0]>0:
                        context_attmaps = []
                        for j in range(context_feature.shape[0]):
                            context_feature_here = context_feature[j]
                            context_feature_here = context_feature_here.repeat(h,w,1) # .cuda()
                            context_attmap = F.cosine_similarity(feature_here,context_feature_here,2).unsqueeze(0).unsqueeze(0)
                            context_attmaps.append(context_attmap.unsqueeze(0))
                        context_attmap = torch.mean(torch.cat(context_attmaps,0),0)
                        att_map = F.relu(att_map - context_attmap)
                    
                    attention_map1 = F.interpolate(att_map, strided_size,mode='bilinear', align_corners=False)[:,0,:,:]
                    attention_map2 = F.interpolate(att_map, strided_up_size,mode='bilinear', align_corners=False)[:,0,:size[0],:size[1]]
                    strided_cam.append(attention_map1.cpu())
                    highres_cam.append(attention_map2.cpu())
                strided_cam = torch.mean(torch.cat(strided_cam,0),0)
                highres_cam = torch.mean(torch.cat(highres_cam,0),0)
                strided_cam = strided_cam/torch.max(strided_cam)                
                highres_cam = highres_cam/torch.max(highres_cam)                
                strided_cams.append(strided_cam.unsqueeze(0))
                highres_cams.append(highres_cam.unsqueeze(0))
            strided_cams = torch.cat(strided_cams,0)
            highres_cams = torch.cat(highres_cams,0)
            np.save(os.path.join(save_path),{"keys": valid_cat,"cam": strided_cams,"high_res": highres_cams})

def run(args):
    ckpt_path = args.cam_weights_name

    if "rs269" in ckpt_path:
        import net.networks

        model = net.networks.Classifier(args.cam_network, 20)
    else:  # Assume RN-50 (IRN's default)
        model = resnet50_cam.Net_CAM()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model.cuda()

    save_feature(model, save_dir=os.path.join(args.work_space,'cam_feature'),voc12_root=args.voc12_root)
    load_feature_select_and_cluster(
        model, workspace=args.work_space,feature_dir=os.path.join(args.work_space,'cam_feature'),mask_dir=args.cam_out_dir,
        select_thres=args.kmeans_select_t, tol=args.kmeans_tol,
        class_thres=0.8, context_thres=0.8,
    )
    make_lpcam(model, workspace=args.work_space,lpcam_out_dir=args.lpcam_out_dir,voc12_root=args.voc12_root,list_name='voc12/train.txt')
