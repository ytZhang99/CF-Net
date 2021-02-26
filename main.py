import torch
from test import Test
from option import args

torch.manual_seed(args.seed)


def main():
    if args.test_only:
        t = Test()
        t.test()
    else:
        from train import Train
        t = Train()
        t.train()


if __name__ == '__main__':
    main()
