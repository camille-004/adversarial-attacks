import click

import src.logger as logger

log = logger.setup_module_level_logger(__name__, file_name="data.log")  # type: ignore


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """
    Run data processing scripts to turn raw (../raw) data into processed data to be saved in ../processed

    :param input_filepath: File path of raw data
    :param output_filepath: File path in which to save processed data
    :return:
    """
    log.info("Making final dataset from raw data")

    # Call data processing functions here


if __name__ == "__main__":
    main()
