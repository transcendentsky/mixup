from .config_parse import cfg
import os
import sys
import torch


def save_checkpoints(epochs, model, run, iters=None):
    output_dir = cfg.EXP_DIR
    checkpoint_prefix = cfg.CHECKPOINTS_PREFIX
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = checkpoint_prefix + '_epoch_{:d}'.format(epochs) + 'run' + str(run) + '.pth'

    filename = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), filename)
    with open(os.path.join(output_dir, 'run' + str(run) + '_checkpoint_list.txt'), 'a') as f:
        f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
    print('Wrote snapshot to: {:s}'.format(filename))

    # TODO: write relative cfg under the same page


def resume_checkpoint(resume_checkpoint, model, run):
    output_dir = cfg.EXP_DIR
    if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
        print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
        return False
    print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
    checkpoint = torch.load(resume_checkpoint)

    # print("=> Weigths in the checkpoints:")
    # print([k for k, v in list(checkpoint.items())])

    # remove the module in the parrallel model
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict

    resume_scope = cfg.TRAIN.RESUME_SCOPE
    # extract the weights based on the resume scope
    if resume_scope != '':
        pretrained_dict = {}
        for k, v in list(checkpoint.items()):
            for resume_key in resume_scope.split(','):
                if resume_key in k:
                    pretrained_dict[k] = v
                    break
        checkpoint = pretrained_dict

    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    # print("=> Resume weigths:")
    # print([k for k, v in list(pretrained_dict.items())])

    checkpoint = model.state_dict()

    unresume_dict = set(checkpoint) - set(pretrained_dict)
    if len(unresume_dict) != 0:
        print("=> UNResume weigths:")
        print(unresume_dict)

    checkpoint.update(pretrained_dict)

    return model.load_state_dict(checkpoint)


def find_previous(run):
    output_dir = cfg.EXP_DIR

    if not os.path.exists(os.path.join(output_dir, 'run' + str(run) + '_checkpoint_list.txt')):
        return False
    with open(os.path.join(output_dir, 'run' + str(run) + '_checkpoint_list.txt'), 'r') as f:
        lineList = f.readlines()
    epoches, resume_checkpoints = [list() for _ in range(2)]
    for line in lineList:
        print("line:", line.find('epoch '), line.find(':'))
        epoch = int(line[line.find('epoch ') + len('epoch '):line.find(':')])
        checkpoint = line[line.find(':') + 2:-1]
        epoches.append(epoch)
        resume_checkpoints.append(checkpoint)
    return epoches, resume_checkpoints
