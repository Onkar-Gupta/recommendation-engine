"""The rec-engine command line tool.

    rec-engine init                  write a config.yaml you can edit
    rec-engine download-data         fetch MovieLens 100K
    rec-engine train --data my.csv   train a model (needs the [train] extra)
    rec-engine serve                 start the API and web page
"""

import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"


def _version():
    try:
        return version("rec-engine")
    except PackageNotFoundError:  # running from a source checkout that was never installed
        return "unknown"


def resolve_config(path=None):
    """The config to use: the one you name, else ./config.yaml, else the built-in default."""
    if path:
        return Path(path)
    local = Path("config.yaml")
    return local if local.is_file() else DEFAULT_CONFIG


def _init(args):
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"{target} already exists. Use --force to overwrite it.", file=sys.stderr)
        return 1
    shutil.copyfile(DEFAULT_CONFIG, target)
    print(f"Wrote {target}. Edit data.columns to match your CSV.")
    return 0


def _download(args):
    from rec_engine.download import download

    download(args.dir)
    return 0


def _train(args):
    try:
        from rec_engine.train import train
    except ImportError:
        print("Training needs TensorFlow. Install it with:  pip install 'rec-engine[train]'", file=sys.stderr)
        return 1
    config = resolve_config(args.config)
    metrics = train(config_path=str(config), data_path=args.data)
    print(
        f"Trained. precision@k={metrics['precision_at_k']:.3f} recall@k={metrics['recall_at_k']:.3f} "
        f"ndcg@k={metrics['ndcg_at_k']:.3f} coverage={metrics['catalogue_coverage']:.1%}"
    )
    return 0


def _serve(args):
    import uvicorn

    from rec_engine.data_loader import load_config
    from rec_engine.serve import create_app

    config = load_config(resolve_config(args.config))
    if args.ratings:
        config["serving"]["ratings_path"] = args.ratings
        # Titles and genres come from MovieLens, so custom data gets plain "Item 123" labels.
        config["serving"]["item_titles_path"] = args.titles or ""
    elif args.titles:
        config["serving"]["item_titles_path"] = args.titles
    try:
        app = create_app(config=config)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="rec-engine", description="Train and serve a recommender.")
    parser.add_argument("--version", action="version", version=f"rec-engine {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write a config.yaml to edit")
    init.add_argument("--path", default="config.yaml")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(run=_init)

    download = commands.add_parser("download-data", help="download MovieLens 100K")
    download.add_argument("--dir", default="data/raw")
    download.set_defaults(run=_download)

    train = commands.add_parser("train", help="train a model and save it to models/")
    train.add_argument("--data", help="your ratings CSV (default: MovieLens 100K)")
    train.add_argument("--config", help="config file (default: ./config.yaml, else the built-in one)")
    train.set_defaults(run=_train)

    serve = commands.add_parser("serve", help="start the API and web page")
    serve.add_argument("--config", help="config file (default: ./config.yaml, else the built-in one)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--ratings", help="the ratings CSV the model was trained on (for custom data)")
    serve.add_argument("--titles", help="optional MovieLens-style u.item file with film titles")
    serve.set_defaults(run=_serve)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
