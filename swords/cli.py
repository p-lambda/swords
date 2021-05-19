if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    from .assets import main as assets_main

    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["assets"])

    argv = sys.argv[1:]
    args = parser.parse_args(argv[:1])

    argv = argv[1:]
    if args.command == "assets":
        assets_main(argv)
    else:
        raise ValueError()
