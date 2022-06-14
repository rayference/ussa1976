"""Command-line interface."""
import click
import numpy as np

from .core import compute


@click.command()
@click.option(
    "--zstart",
    "-z",
    help="Start altitude [m]",
    default=0.0,
    show_default=True,
    type=float,
)
@click.option(
    "--zstop",
    "-Z",
    help="Stop altitude [m]",
    default=1000000.0,
    show_default=True,
    type=float,
)
@click.option(
    "--znum",
    "-n",
    help="Number of altitude points",
    default=1001,
    show_default=True,
    type=int,
)
@click.option(
    "--filename",
    "-f",
    help="Output file name",
    default="ussa1976.nc",
    show_default=True,
    type=click.Path(writable=True),
)
@click.version_option()
def main(
    zstart: float,
    zstop: float,
    znum: int,
    filename: str,
) -> None:
    """Compute the U.S. Standard Atmosphere 1976."""
    click.echo(
        f"Computing U.S. Standard Atmosphere 1976 between {zstart} and "
        f"{zstop} meter on {znum} altitude points."
    )
    z = np.linspace(zstart, zstop, znum)
    ds = compute(z=z)
    click.echo(f"Writing results in {filename}")
    ds.to_netcdf(filename)


if __name__ == "__main__":
    main(prog_name="ussa1976")  # pragma: no cover
