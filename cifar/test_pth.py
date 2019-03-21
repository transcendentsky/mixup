import torch


def main():
    resume_checkpoint1 = "results/baseline/mixup/cifar10/VGG19_cifar10_run_1002_epoch_240_.pth"
    resume_checkpoint2 = "results/baseline/sgd/cifar10/VGG19_cifar10_run_1002_epoch_240_.pth"

    checkpoint1 = torch.load(resume_checkpoint1)
    checkpoint2 = torch.load(resume_checkpoint2)

    def find_test(k):
        idx = k.find('classifier')
        print('str {} , idx = {}'.format(k,idx))
        if idx >= 0:
            return False
        else:return True

    count = 0

    ######     Test Pytorch Ckpt    ######
    print("""
    model.state_dict(): {}
    """.format(0))
    # for k in checkpoint1.keys():
    #     print(k)

    count_less = 0
    count_great = 0
    count_eq = 0

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    count_9 = 0
    print("-------------------------------------------")
    for k, v in checkpoint1.items():
        if k.find('bias') >=0 :
            params1 = checkpoint1[k]
            params2 = checkpoint2[k]
            count_less  += torch.lt(params2, params1).sum().item()
            count_great += torch.gt(params2, params1).sum().item()
            count_eq    += torch.eq(params2, params1).sum().item()

            print("Sum : ", k, torch.lt(params2.sum(), params1.sum()).sum().item(),
                  torch.lt(params2.abs().sum(), params1.abs().sum()).sum().item(), params2.sum(), params1.sum())
            count_1 += torch.lt(params2.sum(), params1.sum()).sum().item()
            count_2 += torch.gt(params2.sum(), params1.sum()).sum().item()
            count_3 += torch.eq(params2.sum(), params1.sum()).sum().item()

            # print("Sum : ", k, torch.lt(params2.abs().sum(), params1.abs().sum()).sum().item(), params2.sum(), params1.sum())
            count_4 += torch.lt(params2.abs(), params1.abs()).sum().item()
            count_5 += torch.gt(params2.abs(), params1.abs()).sum().item()
            count_6 += torch.eq(params2.abs(), params1.abs()).sum().item()

            count_7 += torch.lt(params2.abs().sum(), params1.abs().sum()).sum().item()
            count_8 += torch.gt(params2.abs().sum(), params1.abs().sum()).sum().item()
            count_9 += torch.eq(params2.abs().sum(), params1.abs().sum()).sum().item()

    print("-------------------------------------------")
    print("Count: \n"
          "less {} \n"
          "greater {}\n"
          "eq {}".format(count_less, count_great, count_eq))

    print("\nCount sum(): \n"
          "less {} \n"
          "greater {}\n"
          "eq {}".format(count_1, count_2, count_3))

    print("\nCount abs(): \n"
          "less {} \n"
          "greater {}\n"
          "eq {}".format(count_4, count_5, count_6))

    print("\nCount abs sum(): \n"
          "less {} \n"
          "greater {}\n"
          "eq {}".format(count_7, count_8, count_9))

    # # print("=> Resume weigths:")
    # # print([k for k, v in list(pretrained_dict.items())])
    #
    # checkpoint = model.state_dict()
    #
    # unresume_dict = set(checkpoint) - set(pretrained_dict)
    # if len(unresume_dict) != 0:
    #     print("=> UNResume weigths:")
    #     print(unresume_dict)

    # Some QAQ here ????
    # checkpoint.update(pretrained_dict)
    # for x in  ['module.classifier.weight', 'module.classifier.bias']:
    #     checkpoint.popitem(x)

    # print(type(checkpoint))
    # print(checkpoint.keys())
    #
    # return model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint)

if __name__ == '__main__':
    main()