import argparse
import json

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "nonprehensile"


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bank = sub.add_parser("bank")
    bank.add_argument("--output", required=True)
    bank.add_argument("--count", type=int, default=8192)
    bank.add_argument("--seed", type=int, default=101)
    bank.add_argument("--workers", type=int, default=8)
    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--root", default="runs/experiment")
    pipeline.add_argument("--gpus", type=int, nargs="+", default=[0])
    pipeline.add_argument("--num-envs", type=int, default=4096)
    pipeline.add_argument("--search-iterations", type=int, default=300)
    pipeline.add_argument("--final-iterations", type=int, default=2000)
    pipeline.add_argument("--bank-size", type=int, default=8192)
    pipeline.add_argument("--workers", type=int, default=8)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--bank", required=True)
    train.add_argument("--validation-bank", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda:0")
    test = sub.add_parser("test")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--bank", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--episodes", type=int, default=8192)
    test.add_argument("--device", default="cuda:0")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "bank":
        from .bank import build

        build(args.pop("output"), **args)
    elif command == "pipeline":
        from .pipeline import run

        run(**args)
    elif command == "train":
        from .ppo import train

        with open(args.pop("config")) as file:
            config = json.load(file)
        train(config, args.pop("bank"), args.pop("validation_bank"), **args)
    else:
        from .ppo import test

        test(**args)


if __name__ == "__main__":
    main()
