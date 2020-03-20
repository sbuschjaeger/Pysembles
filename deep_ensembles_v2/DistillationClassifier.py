import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import tqdm
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from .Utils import TransformTensorDataset
from .Models import SKLearnModel

class DistillationModel(SKLearnModel):
    def __init__(self, distillation_epochs=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_epochs = distillation_epochs

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)
        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        y_tensor = y_tensor.type(torch.LongTensor)

        assert self.transformer is None
        if sample_weight is not None:
            sample_weight = len(y)*sample_weight/np.sum(sample_weight)
            w_tensor = torch.tensor(sample_weight)
            w_tensor = w_tensor.type(torch.FloatTensor)
            data = TransformTensorDataset(x_tensor, y_tensor, w_tensor, transform=self.transformer)
        else:
            w_tensor = None
            data = TransformTensorDataset(x_tensor, y_tensor, transform=self.transformer)

        self.X_ = X
        self.y_ = y

        optimizer = self.optimizer_method(self.parameters(), **self.optimizer)
        assert isinstance(self.loss_function, nn.CrossEntropyLoss)
        distillation_loss = nn.KLDivLoss(reduction="none")
        if self.scheduler_method is not None:
            scheduler = self.scheduler_method(optimizer, **self.scheduler)
        else:
            scheduler = None

        cuda_cfg = {'num_workers': 1, 'pin_memory': True}

        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            **cuda_cfg
        )

        self.cuda()
        self.train()

        if self.out_path is not None:
            outfile = open(self.out_path + "/" + self.training_csv, "w", 1)
            if self.x_test is not None:
                o_str = "epoch,train-loss,train-accuracy,test-loss,test-accuracy"
            else:
                o_str = "epoch,train-loss,train-accuracy"
            outfile.write(o_str + "\n")
        targets = []
        for epoch in range(self.epochs):

            # New Targets!!!
            if epoch > 0 and epoch % self.distillation_epochs == 0:
                newlabels = torch.stack(targets, dim=0)
                train_loader = torch.utils.data.DataLoader(
                    TransformTensorDataset(x_tensor, newlabels, transform=self.transformer),
                            batch_size=self.batch_size, shuffle=False, num_workers=0)
                self.layers_ = self.base_estimator()
                self.cuda()
                self.train()
                optimizer = self.optimizer_method(self.parameters(), **self.optimizer)
                if self.scheduler_method is not None:
                    scheduler = self.scheduler_method(optimizer, **self.scheduler)
                else:
                    scheduler = None

            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0
            with tqdm.tqdm(total=len(train_loader.dataset), ncols=135, disable=not self.verbose) as pbar:
                targets = []
                for batch in train_loader:
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    if sample_weight is not None:
                        weights = batch[2]
                        weights = weights.cuda()
                        weights = Variable(weights)

                    optimizer.zero_grad()
                    output = self(data)
                    for logits in output:
                        targets.append(logits.detach().cpu())
                    if epoch < self.distillation_epochs:
                        unweighted_acc = (output.argmax(1) == target).type(torch.cuda.FloatTensor)
                        if sample_weight is not None:
                            loss = self.loss_function(output, target, weights)
                            epoch_loss += loss.sum().item()
                            loss = loss.mean()

                            weighted_acc = unweighted_acc*weights
                            n_correct += weighted_acc.sum().item()
                        else:
                            loss = self.loss_function(output, target)
                            epoch_loss += loss.sum().item()
                            loss = loss.mean()
                            n_correct += unweighted_acc.sum().item()
                    else:
                        unweighted_acc = (output.argmax(1) == target.argmax(1)).type(torch.cuda.FloatTensor)
                        loss = distillation_loss(
                            nn.functional.log_softmax(output, dim=1),
                            nn.functional.softmax(target, dim=1)
                        )
                        epoch_loss += loss.sum().item()
                        loss = loss.mean()
                        n_correct += unweighted_acc.sum().item()

                    loss.backward()
                    optimizer.step()

                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1
                    # print("")
                    # print(self.layers_[-3].running_var)
                    # print(self.layers_[-3].running_mean)
                    # print("")
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                        epoch,
                        self.epochs-1,
                        epoch_loss/example_cnt,
                        100. * n_correct/example_cnt
                    )
                    pbar.set_description(desc)

                if self.x_test is not None and epoch % self.eval_test == self.eval_test - 1:
                    pred_proba = self.predict_proba(self.x_test)
                    pred_tensor = torch.tensor(pred_proba).cuda()
                    y_test_tensor = torch.tensor(self.y_test).cuda()
                    test_loss = self.loss_function(pred_tensor, y_test_tensor).mean().item()
                    accuracy_test = accuracy_score(self.y_test, np.argmax(pred_proba, axis=1)) * 100.0

                    desc = '[{}/{}] loss {:2.4f} train acc {:2.4f} test loss/acc {:2.4f}/{:2.4f}'.format(
                        epoch,
                        self.epochs-1,
                        epoch_loss/example_cnt,
                        100. * n_correct/example_cnt,
                        test_loss,
                        accuracy_test
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()

            if self.out_path is not None:
                avg_loss = epoch_loss/example_cnt
                accuracy = 100.0*n_correct/example_cnt

                if self.x_test is not None:
                    if epoch % self.eval_test != self.eval_test - 1:
                        accuracy_test = "-"
                        test_loss = "-"
                    outfile.write("{},{},{},{},{}\n".format(
                        epoch, avg_loss, accuracy, test_loss, accuracy_test
                    ))
                else:
                    outfile.write("{},{},{}\n".format(epoch, avg_loss, accuracy))

    def forward(self, x):
        return self.layers_(x)
