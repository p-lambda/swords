if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    from .assets import main as assets_main
    from .datasets import main as parse_dataset_main
    from .eval import main as eval_main

    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["assets", "parse", "eval"])

    argv = sys.argv[1:]
    args = parser.parse_args(argv[:1])

    argv = argv[1:]
    if args.command == "assets":
        assets_main(argv)
    elif args.command == "parse":
        parse_dataset_main(argv)
    elif args.command == "eval":
        eval_main(argv)
    else:
        raise ValueError()
