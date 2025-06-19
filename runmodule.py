import copy
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
import evaluation

from torch.utils.data import DataLoader, TensorDataset
from datasets import get_loader,get_loader_recover
from dataloader import load_data
from model import get_model
from model_con import Network
from loss import TransformerLoss, Loss
from sklearn.preprocessing import StandardScaler
from util import getMvKNNGraph,getMvKNNGraph_hs
from metric import valid
from util import plot_tsne


class RunModule:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device

        # get data
        self.loader, self.features, self.labels, self.inc_idx, self.masked_x, self.ss_list = get_loader(self.cfg, self.device)
        
        self.model_path = "./pretrain/" + self.cfg['Dataset']["name"]
        self.data_recover_path = "./data/recover_data/" + self.cfg['Dataset']["name"]
    
    def recover_train(self):
        epochs = self.cfg['training']['epoch_em']
        num_views = self.cfg['Dataset']['num_views']
        print_num = self.cfg['print_num']

        loader = self.loader
        masked_x = [torch.from_numpy(x).to(self.device) for x in self.masked_x]
        # get model
        module = get_model(self.cfg['Module']['in_dim'], d_model=self.cfg['Module']['trans_dim'],
                           n_layers=self.cfg['Module']['trans_layers'], heads=self.cfg['Module']['trans_headers'],
                           classes_num=self.cfg['Dataset']['num_classes'], dropout=self.cfg['Module']['trans_dropout'],
                           device=self.device)
        loss_model = TransformerLoss()
        optimizer = torch.optim.Adam(module.parameters(), lr=self.cfg['training']['lr'])

        module.train()
        for epoch in range(epochs):
            for x_list, inc, idx in loader:
                """
                encX:Rawdata -> embeddinglayers -> ETrans
                decX:Rawdata -> embeddinglayers -> ETrans -> Fusion -> DTrans
                x_bar:Rawdata -> embeddinglayers -> ETrans -> Fusion -> DTrans -> re_embeddinglayers
                h:Rawdata -> embeddinglayers -> ETrans -> Fusion 
                """
                encX,decX,x_bar,h,_ = module.forward_refactor(copy.deepcopy(x_list),mask = inc,recover=True)
                mse_loss = loss_model.weighted_wmse_loss(x_bar, x_list, inc)
                
                loss = mse_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % print_num == 0:               
                
                if self.cfg['Dataset']['num_sample']>=20000:
                    h = torch.zeros((self.cfg['Dataset']['num_sample'], 512), device=self.device)
                    for x_list, inc, idx in loader:
                        with torch.no_grad():
                            _,_,_,h_batch,_ = module.forward_refactor(copy.deepcopy(x_list),mask = inc,recover=True)
                            h[idx] = h_batch
                    res = evaluation.clustering([h.detach().cpu().numpy()], self.labels)
                    print(str(epoch) + ': kmeans on H' + str(res['kmeans']) + "\033[0m")
                    
                else:
                    with torch.no_grad():
                        _,_,_,h,_ = module.forward_refactor(copy.deepcopy(masked_x),torch.from_numpy(self.inc_idx).to(self.device),recover=True)
                        res = evaluation.clustering([h.detach().cpu().numpy()], self.labels)
                        print(str(epoch) + ': kmeans on H' + str(res['kmeans']) + "\033[0m")

            if epoch == epochs - 1:
                
                if self.cfg['Dataset']['num_sample']>=20000:
                    x_bar = [torch.zeros((self.cfg['Dataset']['num_sample'], in_dim), 
                                         device=self.device) for in_dim in self.cfg['Module']['in_dim']]
                    h = torch.zeros((self.cfg['Dataset']['num_sample'], 512), device=self.device)
                    for x_list, inc, idx in loader:
                        with torch.no_grad():
                            _,_,x_bar_batch,h_batch,_ = module.forward_refactor(copy.deepcopy(x_list),mask = inc,recover=True)
                            h[idx] = h_batch
                            for v in range(num_views):
                                x_bar[v][idx,:] = x_bar_batch[v]
                
                else:
                    _,_,x_bar,h,_ = module.forward_refactor(copy.deepcopy(masked_x), torch.from_numpy(self.inc_idx).to(self.device),recover=True)
        
        x_recover = []

        for v in range(len(masked_x)):
            original_data = masked_x[v]  
            recovered_data = x_bar[v]   

            reconstructed_view = original_data.clone() 
            reconstructed_view [(self.inc_idx[:, v] == 0)] = recovered_data [(self.inc_idx[:, v] == 0)]  # 对于缺失的样本，使用恢复的视图数据
            x_recover.append(reconstructed_view)
        
        #将x_recover反归一化为原始数据
        x_recover_original = []
        for v, recovered_data in enumerate(x_recover):
            recovered_data_cpu = recovered_data.detach().cpu().numpy() 
            original_data_recovered = self.ss_list[v].inverse_transform(recovered_data_cpu)  # 转换为原始数据
            original_data_recovered_tensor = torch.from_numpy(original_data_recovered).float()
            x_recover_original.append(original_data_recovered_tensor)

        x_recover = x_recover_original
        
        #保存模型
        torch.save({'model': module.state_dict()}, self.model_path + '_Rec.pth')
        
        # save data.mat
        labels = self.labels    
        x_list = [x.numpy() for x in x_recover]
        data_dict = {}
        for i, x in enumerate(x_list):
            data_dict[f'X{i+1}'] = x.astype(np.float32) 
        data_dict['Y'] = labels.astype(np.int32)
        scipy.io.savemat(self.data_recover_path + '_recover.mat', data_dict)

        
    
    def contrastive_train(self):
        
        mse_epochs = self.cfg['training']['mse_epochs']
        con_epochs = self.cfg['training']['con_epochs']
        num_views = self.cfg['Dataset']['num_views']
        print_num = self.cfg['print_num']
        batch_size = self.cfg['Dataset']['batch_size']
        temperature_f = self.cfg['temperature_f']
        feature_dim = self.cfg['feature_dim']
        high_feature_dim = self.cfg['high_feature_dim']
    
        
        # load recovered data
        dataset, dims, view, data_size, class_num = load_data(self.data_recover_path,self.cfg['Dataset']['name'])
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        if self.cfg['Dataset']['name'] in ['HandWritten']:
            # HandWritten
            view_data_list = []
            view_data_list.append(torch.from_numpy(dataset.view1))
            view_data_list.append(torch.from_numpy(dataset.view2))
            view_data_list.append(torch.from_numpy(dataset.view3))
            view_data_list.append(torch.from_numpy(dataset.view4))
            view_data_list.append(torch.from_numpy(dataset.view5))
            
        elif self.cfg['Dataset']['name'] in ['Fashion']:
            # Fashion
            view_data_list = []
            view_data_list.append(torch.from_numpy(dataset.V1))
            view_data_list.append(torch.from_numpy(dataset.V2))
            view_data_list.append(torch.from_numpy(dataset.V3))
        
        elif self.cfg['Dataset']['name'] in ['Scene-15','aloideep3v', 'CIFAR10']:
            # Scene, aloideep3v, CIFAR10
            view_data_list = []
            view_data_list.append(torch.from_numpy(dataset.view1))
            view_data_list.append(torch.from_numpy(dataset.view2))
            view_data_list.append(torch.from_numpy(dataset.view3))

        else:
            # Caltech101-7
            view_data_list = []
            view_data_list.append(torch.from_numpy(dataset.view1))
            view_data_list.append(torch.from_numpy(dataset.view2))
            view_data_list.append(torch.from_numpy(dataset.view3))
            view_data_list.append(torch.from_numpy(dataset.view4))
            view_data_list.append(torch.from_numpy(dataset.view5))
            view_data_list.append(torch.from_numpy(dataset.view6))
        
        # get model
        module_con = Network(view, dims, feature_dim, high_feature_dim, class_num, self.device)
    
        module_con = module_con.to(self.device)
        optimizer = torch.optim.Adam(module_con.parameters(), lr=self.cfg['training']['lr_con'], weight_decay=0.)
        criterion = Loss(batch_size, num_views, temperature_f, self.device).to(self.device)
        
        
        all_new_x = [torch.from_numpy(np.array(v_data)) for v_data in view_data_list]
        all_new_z = torch.ones((view, data_size, self.cfg['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, self.cfg['training']['knn']), device=self.device,
                                        dtype=torch.float32)
        
        all_new_x = [x.to(self.device) for x in all_new_x]


        # Pretrain
        for epoch in range(mse_epochs):
            tot_loss = 0.
            criterion = torch.nn.MSELoss()
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                for v in range(view):
                    xs[v] = xs[v].to(self.device)
                optimizer.zero_grad()
                _, xrs, _ = module_con(xs)
                loss_list = []
                for v in range(view):
                    loss_list.append(criterion(xs[v], xrs[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            # print('Epoch {}'.format(epoch + 1), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        
        criterion = Loss(batch_size, num_views, temperature_f, self.device).to(self.device)
    
        # contrastive train
        for epoch in range(con_epochs):
            tot_loss = 0.
            mes = torch.nn.MSELoss()
            for xs, _, idx in data_loader:
                for v in range(view):
                    xs[v] = xs[v].to(self.device)
                optimizer.zero_grad()
                hs, xrs, zs = module_con(xs)
                loss_list = []
                for v in range(view):
                    for w in range(v+1, view):
                        loss_list.append(criterion.forward_feature(hs[v], hs[w]) * self.cfg['training']['loss_weight1'])
                    loss_list.append(mes(xs[v], xrs[v]) * self.cfg['training']['loss_weight2'])
                loss = sum(loss_list)
                
                if epoch > 0:
                    loss = loss + self.cfg['training']['lambda_graph'] * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
                for v in range(view):
                    all_new_z[v][idx] = zs[v].detach().clone()
                    # all_new_hs[v][idx] = hs[v].detach().clone()
            
            
            # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
            # acc, nmi, pur, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=True, eval_h=True)

            
            if epoch == con_epochs - 1:
                print("Epoch: ", epoch)
                acc, nmi, pur, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=True, eval_h=True)


        # save model
        torch.save({'model': module_con.state_dict()}, self.model_path + '_Con.pth')

                    
                    
    def SelfPaced_train(self):
        
        epoch1 = self.cfg['training']['epoch1']
        epoch2 = self.cfg['training']['epoch2']
        num_views = self.cfg['Dataset']['num_views']
        batch_size = self.cfg['Dataset']['batch_size']
        feature_dim = self.cfg['feature_dim']
        high_feature_dim = self.cfg['high_feature_dim']
        temperature_f = self.cfg['temperature_f']
        
        
        for k in range(3):
            print('Cycle {} starts:'.format(k + 1))
           
            # load contrastive model 
            dataset, dims, view, data_size, class_num = load_data(self.data_recover_path,self.cfg['Dataset']['name'])
            module_con = Network(view, dims, feature_dim, high_feature_dim, class_num, self.device)
            checkpoint = torch.load(self.model_path + '_Con.pth', map_location=self.device)
            module_con.load_state_dict(checkpoint['model']) 
            module_con = module_con.to(self.device)
            
            _, _, _, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=False, eval_h=True)
            all_new_hs = np.mean(high_level_vectors, axis=0)
            
            # Calculate self-paced weight
            weights = torch.zeros(data_size, len(high_level_vectors)) 
            all_new_hs_tensor = torch.from_numpy(all_new_hs).float()
            all_new_norm = F.normalize(all_new_hs_tensor, p=2, dim=1)
            for i, vectors in enumerate(high_level_vectors):
                view_tensor = torch.from_numpy(vectors).float()
                view_norm = F.normalize(view_tensor, p=2, dim=1)
                cosine_similarity = (view_norm * all_new_norm).sum(dim=1) 
                weights[:, i] = cosine_similarity

            # inc_idx = torch.tensor(self.inc_idx, dtype=torch.bool)
            # weights[inc_idx] = 1
            
            # load recover model
            loader, features, labels, inc_idx, masked_x, ss_list = get_loader_recover(self.cfg, self.device)
            masked_x = [torch.from_numpy(x).to(self.device) for x in self.masked_x]
            x_eval = [torch.from_numpy(view).to(self.device) for view in features]
            all_new_x = [torch.from_numpy(view).to(self.device) for view in features]
        
            module = get_model(self.cfg['Module']['in_dim'], d_model=self.cfg['Module']['trans_dim'],
                            n_layers=self.cfg['Module']['trans_layers'], heads=self.cfg['Module']['trans_headers'],
                            classes_num=self.cfg['Dataset']['num_classes'], dropout=self.cfg['Module']['trans_dropout'],
                            load_weights=self.model_path + '_Rec.pth',device=self.device)
            
            loss_model = TransformerLoss()
            optimizer = torch.optim.Adam(module.parameters(), lr=self.cfg['training']['lr'])
            criterion = Loss(batch_size, num_views, temperature_f, self.device).to(self.device)
            module.train()
            
            print('Recover module begin:')
            for epoch in range(epoch1):
                for x_list, inc, idx in loader:
                    
                    tot_loss = 0.
                    encX,decX,x_bar,h,_ = module.forward_refactor(copy.deepcopy(x_list),mask = None,recover = False)
                    loss = loss_model.selfpaecd_wmse_loss(x_bar, x_list, inc, weights[idx], device = self.device)
                    # loss = loss_model.weighted_wmse_loss(x_bar, x_list, inc)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tot_loss += loss.item()

                    for v in range(view):
                        all_new_x[v][idx] = x_bar[v].detach().clone()
                
                if epoch % 10 == 0:
                    with torch.no_grad():
                        module.eval()
                        _,_,_,h,_ = module.forward_refactor(copy.deepcopy(x_eval),mask = None,recover = False)
                        res = evaluation.clustering([h.detach().cpu().numpy()], self.labels)
                        print(str(epoch) + ': kmeans on H' + str(res['kmeans']) + "\033[0m")
                        module.train()

            _,_,x_bar,h,_ = module.forward_refactor(copy.deepcopy(x_eval), mask = None,recover=False)
            
            x_recover = []

            for v in range(len(masked_x)):
                original_data = masked_x[v]  
                recovered_data = x_bar[v]   

                reconstructed_view = original_data.clone() 
                reconstructed_view [(self.inc_idx[:, v] == 0)] = recovered_data [(self.inc_idx[:, v] == 0)]  # 对于缺失的样本，使用恢复的视图数据
                x_recover.append(reconstructed_view)
            
            #将x_recover反归一化为原始数据
            x_recover_original = []
            for v, recovered_data in enumerate(x_recover):
                recovered_data_cpu = recovered_data.detach().cpu().numpy() 
                original_data_recovered = self.ss_list[v].inverse_transform(recovered_data_cpu)  # 转换为原始数据
                original_data_recovered_tensor = torch.from_numpy(original_data_recovered).float()
                x_recover_original.append(original_data_recovered_tensor)

            x_recover = x_recover_original    
            
            # save data_sp.mat and model
            labels = self.labels    
            x_list = [x.numpy() for x in x_recover]
            data_dict = {}
            for i, x in enumerate(x_list):
                data_dict[f'X{i+1}'] = x.astype(np.float32) 
            data_dict['Y'] = labels.astype(np.int32)
            scipy.io.savemat(self.data_recover_path + '_recover.mat', data_dict)
            
            torch.save({'model': module.state_dict()}, self.model_path + '_Rec.pth')
            

            dataset, dims, view, data_size, class_num = load_data(self.data_recover_path,self.cfg['Dataset']['name'])
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            
            if self.cfg['Dataset']['name'] in ['HandWritten']:
                # HandWritten
                view_data_list = []
                view_data_list.append(torch.from_numpy(dataset.view1))
                view_data_list.append(torch.from_numpy(dataset.view2))
                view_data_list.append(torch.from_numpy(dataset.view3))
                view_data_list.append(torch.from_numpy(dataset.view4))
                view_data_list.append(torch.from_numpy(dataset.view5))
                
            elif self.cfg['Dataset']['name'] in ['Fashion']:
                # Fashion
                view_data_list = []
                view_data_list.append(torch.from_numpy(dataset.V1))
                view_data_list.append(torch.from_numpy(dataset.V2))
                view_data_list.append(torch.from_numpy(dataset.V3))
            
            elif self.cfg['Dataset']['name'] in ['Scene-15','aloideep3v', 'CIFAR10']:
                # Scene, aloideep3v
                view_data_list = []
                view_data_list.append(torch.from_numpy(dataset.view1))
                view_data_list.append(torch.from_numpy(dataset.view2))
                view_data_list.append(torch.from_numpy(dataset.view3))
            
            else:
                # Caltech101-7
                view_data_list = []
                view_data_list.append(torch.from_numpy(dataset.view1))
                view_data_list.append(torch.from_numpy(dataset.view2))
                view_data_list.append(torch.from_numpy(dataset.view3))
                view_data_list.append(torch.from_numpy(dataset.view4))
                view_data_list.append(torch.from_numpy(dataset.view5))
                view_data_list.append(torch.from_numpy(dataset.view6))
            
            optimizer = torch.optim.Adam(module_con.parameters(), lr=self.cfg['training']['lr_con'], weight_decay=0.)
            criterion = Loss(batch_size, num_views, temperature_f, self.device).to(self.device)
            
            all_new_xcon = [torch.from_numpy(np.array(v_data)) for v_data in view_data_list]
            all_new_z = torch.ones((view, data_size, self.cfg['feature_dim'])).to(self.device)
            all_graph = torch.tensor(getMvKNNGraph(all_new_xcon, self.cfg['training']['knn']), device=self.device,
                                            dtype=torch.float32)
            
            all_new_xcon = [x.to(self.device) for x in all_new_xcon]
            
            print('Contrastive module begin:')
            for epoch in range(epoch2):
                tot_loss = 0.
                mes = torch.nn.MSELoss(reduction="none")
                for xs, _, idx in data_loader:
                    for v in range(view):
                        xs[v] = xs[v].to(self.device)
                    optimizer.zero_grad()
                    hs, xrs, zs = module_con(xs)
                    loss_list = []
                    for v in range(view):
                        for w in range(v+1, view):
                            # loss_list.append(criterion.forward_feature_selfpaced(hs[v], hs[w], weights[idx][:, v]) * self.cfg['training']['loss_weight1'])
                            loss_list.append(criterion.forward_feature(hs[v], hs[w]) * self.cfg['training']['loss_weight1'])
                        rows, cols = mes(xs[v], xrs[v]).shape
                        mes_row = torch.sum(mes(xs[v], xrs[v]), dim=1, keepdim=True)
                        weights_idx = weights[idx][:, v].unsqueeze(1).to(self.device)
                        # self_paced *
                        loss_xs = torch.sum(weights_idx * mes_row) / (rows * cols)
                        loss_list.append(loss_xs)
                    loss = sum(loss_list)
                    
                    if epoch > 0:
                        loss = loss + self.cfg['training']['lambda_graph'] * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)
                    loss.backward()
                    optimizer.step()
                    tot_loss += loss.item()
                    for v in range(view):
                        all_new_z[v][idx] = zs[v].detach().clone()
                
                
                # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
                # acc, nmi, pur, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=True, eval_h=True)
                # if epoch % 10 == 0:
                #     acc, nmi, pur, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=True, eval_h=True)
                
                if epoch == epoch2 - 1:
                    acc, nmi, pur, high_level_vectors = valid(module_con, self.device, dataset, view, data_size, class_num, flag=True, eval_h=True)
            
            """
            # t-NSE

            if k == 0:
                # average_vector = np.average(high_level_vectors, axis=0)
                concatenated_vectors = np.concatenate(high_level_vectors, axis=1)
                plot_tsne(concatenated_vectors, labels)
            
            """


            # save model
            torch.save({'model': module_con.state_dict()}, self.model_path + '_Con.pth')
            
            print('Cycle {} end'.format(k + 1))