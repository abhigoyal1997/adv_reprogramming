import torch
import torch.optim as optim


def save_checkpoint(epoch, program, mask, optimizer, best_val, lr_scheduler, file_path='models/checkpoint'):
    torch.save({
        'epoch': epoch,
        'program': program,
        'mask': mask,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_val': best_val
    }, file_path)
    print('Checkpoint created: {}'.format(file_path))


def save_model(program, mask=None, file_path='models/model'):
    if mask is not None:
        torch.save({
            'program': program,
            'mask': mask,
        }, file_path)
    else:
        torch.save({
            'program': program,
        }, file_path)
    print('Model saved: {}'.format(file_path))


def load_checkpoint(optimizer=None, lr_scheduler=None, file_path='models/checkpoint'):
    state = torch.load(file_path)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state['lr_scheduler'])
    print('Checkpoint restored: {}'.format(file_path))
    return state['program'], state['epoch'], state['best_val']


class LRScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, patience=2, verbose=True, factor=0.96, mode='max'):
        super(LRScheduler, self).__init__(optimizer, patience=patience, verbose=verbose, factor=factor, min_lr=1.6e-6, mode=mode)

    def is_impatient(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            # self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True
        return False

    def reduce_lr(self, epoch=None):
        reduced = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                reduced = True
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
        return reduced
