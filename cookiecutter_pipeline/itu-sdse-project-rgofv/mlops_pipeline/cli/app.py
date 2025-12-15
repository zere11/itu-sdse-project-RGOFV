import typer
from pathlib import Path
from loguru import logger


from mlops_pipeline.data.prepare import load_and_prepare_data
from mlops_pipeline.features_cli import feature_app
from mlops_pipeline.dataset import data_app
from mlops_pipeline.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, PRINTING_STATE

app = typer.Typer(help="Hi, and welcome to the RGoFV MLOps app! We are happy to see you here. \n This is a combined MLOps pipeline CLI. Commands are listed in order of operation.")
app.add_typer(data_app, name="data")
app.add_typer(feature_app, name="features")

@app.command("prepare")
def prepare(
    data_gold_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "training_features_gold.csv", 
        help="Sets the path for finding the gold data, as created by the build command. Default is processed/training_features_gold.csv" ),
    data_out_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        help="Directory to write the X/y train/test to. Defaults to data/interim.",
    ),
    testing_size: float = typer.Option(
        0.15,
        help="Sets the size of the test set. Default is 15% (0.15).",
    ),
    random_state_set: int = typer.Option(
        42,
        help="Sets the random state for the split. Default is 42.",
    ),
    printing_state: bool = typer.Option(
        PRINTING_STATE,
        help="Sets the printing state for the script. If set to True, prints as the script runs for troubleshooting.",
    ),
):
    '''
    Runs the data separation scripts in data/prepare.py
    '''
    logger.info("Making the test/train split...")
    load_and_prepare_data(
        data_gold_path=data_gold_path,
        out_dir=data_out_dir,
        test_size=testing_size,
        random_state=random_state_set,
        printing=printing_state,
    )
    logger.success(f"Split train/test-files have been created at {data_out_dir}.")

if __name__ == "__main__":
    app()