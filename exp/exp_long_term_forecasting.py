from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import warnings
import numpy as np
import random

warnings.filterwarnings('ignore')


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (index, *self.dataset[index])


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.timefilter_masks = self._get_timefilter_mask() if self.args.model == 'TimeFilter' else None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.model == 'RAFT':
            train_data, _ = self._get_data(flag='train')
            vali_data, _ = self._get_data(flag='val')
            test_data, _ = self._get_data(flag='test')
            model.prepare_dataset(train_data, vali_data, test_data)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        if self.args.model == 'RAFT':
            shuffle_flag = False if flag == 'test' else True
            data_set = IndexedDataset(data_set)
            data_loader = DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=shuffle_flag,
                num_workers=self.args.num_workers,
                drop_last=False)
        return data_set, data_loader

    def _select_optimizer(self):
        wd = getattr(self.args, 'weight_decay', 0.0)
        if wd and wd > 0:
            model_optim = optim.AdamW(self.model.parameters(),
                                      lr=self.args.learning_rate, weight_decay=wd)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_timefilter_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) &
                 (torch.arange(L) < k // N * N + N) &
                 (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        return torch.stack(masks, dim=0)

    def _unpack_batch(self, batch):
        if self.args.model == 'RAFT':
            index, batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return index, batch_x, batch_y, batch_x_mark, batch_y_mark
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return None, batch_x, batch_y, batch_x_mark, batch_y_mark

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, index=None, mode='train'):
        if self.args.model == 'RAFT':
            return self.model(batch_x, index, mode=mode), None
        if self.args.model == 'TimeFilter':
            outputs, moe_loss = self.model(
                batch_x, self.timefilter_masks, is_training=(mode == 'train'))
            return outputs, moe_loss
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0], None
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark), None

    def vali(self, vali_data, vali_loader, criterion, mode='valid'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode=mode)
                else:
                    outputs, _ = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode=mode)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path = path + '/' + 'checkpoint.pth'
        if (os.path.exists(best_model_path) & bool(self.args.load_data)):
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        clip_grad = getattr(self.args, 'clip_grad', 0.0)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            max_memory = 0  # Initialize maximum graphics memory usage
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                if self.args.noise:
                    epsilon=self.args.noise
                    time_points = random.sample(range(batch_x.size()[1]), round(batch_x.size()[1]*epsilon))
                    # Create a tensor with the same shape as the original tensor to store perturbations
                    perturbed_tensor = torch.zeros_like(batch_x)
                    # Add perturbations to each selected time point
                    for time_point in time_points:
                        # Obtain raw data
                        original_data = batch_x[:, time_point, :]
                        # Generate random noise within the range of [-2Xi, 2Xi]
                        noise = torch.rand_like(original_data) * 4 * original_data - 2 * original_data
                        perturbed_tensor[:, time_point, :] += noise.float().to(self.device)
                    batch_x += perturbed_tensor

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, moe_loss = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode='train')

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if moe_loss is not None:
                            loss = loss + 0.05 * moe_loss
                        train_loss.append(loss.item())
                else:
                    outputs, moe_loss = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode='train')

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if moe_loss is not None:
                        loss = loss + 0.05 * moe_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print('\tspeed: {:.4f}s/iter; {:.4f}ms/iter; left time: {:.4f}s'.format(speed, speed * 1000, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if clip_grad and clip_grad > 0:
                        scaler.unscale_(model_optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad and clip_grad > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, mode='valid')
            test_loss = self.vali(test_data, test_loader, criterion, mode='test')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode='test')
                else:
                    outputs, _ = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, index=index, mode='test')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                input_data = input[0, :, -1]
                gt = true[0, :, -1]
                pd = pred[0, :, -1]
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'), input_data=input_data)

        # preds = np.array(preds)
        # trues = np.array(trues)
        preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        trues = np.concatenate(trues, axis=0) # without the "drop-last" trick
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
