import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import glob
import time
from tqdm import tqdm

from global_variables import DATA_DIR, NB_OUTPUT_CLASSES


def cal_acc(logit, label):
    logit = logit.view(-1)
    label = label.view(-1)
    n_sample = len(logit)
    pred = (F.sigmoid(logit) > 0.5).float()

    n_acc = (pred.data == label.data).sum()
    return n_acc, n_sample

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    makedirs(args.save_path)
    print(header)

    def eval():
        # switch model to evaluation mode
        model.eval(); dev_iter.init_epoch()

        # calculate accuracy on validation set
        n_dev_correct, n_dev_total, dev_loss = 0, 0, 0
        for dev_batch_idx, dev_batch in enumerate(dev_iter):
            dev_logit = model(dev_batch.text)
            ndc, ndt = cal_acc(dev_logit, dev_batch.label)
            n_dev_correct += ndc
            n_dev_total += ndt
            dev_loss = criterion(dev_logit, dev_batch.label)
        
        dev_acc = 100. * n_dev_correct / n_dev_total
        return dev_loss, dev_acc

    def save(acc, loss, iter, name='snapshot'):
        snapshot_prefix = os.path.join(args.save_path, name)
        snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(acc, loss.data[0], iter)
        torch.save(model, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)



    for epoch in range(args.epochs):
        train_iter.init_epoch()
        for batch_idx, batch in enumerate(train_iter):
            # switch model to training mode, clear gradient accumulators
            model.train(); opt.zero_grad()
            iterations += 1

            # forward pass
            logit = model(batch.text)

            # calculate accuracy of predictions in the current batch
            n_correct, n_total = cal_acc(logit, batch.label)
            train_acc = 100. * n_correct / n_total

            # calculate loss of the network output with respect to training labels
            loss = criterion(logit, batch.label)

            # backpropagate and update optimizer learning rate
            loss.backward(); opt.step()

            # evaluate performance on validation set periodically
            if (args.dev_every > 0) and (iterations % args.dev_every == 0):
                dev_loss, dev_acc = eval()
                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:
                    # found a model with better validation set accuracy
                    best_dev_acc = dev_acc
                    save(dev_acc, dev_loss, iterations)

            elif iterations % args.log_every == 0:

                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

        # eval and save after an epoch
        dev_loss, dev_acc = eval()
        print(dev_log_template.format(time.time()-start,
            epoch, iterations, 1+batch_idx, len(train_iter),
            100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

        # update best valiation set accuracy
        if dev_acc > best_dev_acc:
            # found a model with better validation set accuracy
            best_dev_acc = dev_acc
            save(dev_acc, dev_loss, iterations)

def predict(test_iter, model, args):
    model.eval()
    submission = pd.DataFrame()
    test_iter.init_epoch()
    for batch_idx, batch in tqdm(enumerate(test_iter)):
        logit = model(batch.text)
        probs = F.sigmoid(logit)
        batch_results_id = pd.DataFrame({'id': data["id"]})
        batch_results = pd.concat([batch_results_id, pd.DataFrame(probs.data, columns = OUTPUT_LABELS)], axis=1)
        submission = pd.concat(submission, batch_results)
    submission.to_csv(args.output, index=False)

    

