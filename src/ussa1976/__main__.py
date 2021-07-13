"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Ussa1976."""


if __name__ == "__main__":
    main(prog_name="ussa1976")  # pragma: no cover
