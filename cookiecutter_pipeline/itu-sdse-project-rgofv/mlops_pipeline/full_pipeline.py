import subprocess
from loguru import logger


def run_full_pipeline():
    commands = [
        ["python", "-m", "mlops_pipeline.cli.app", "data"],
        ["python", "-m", "mlops_pipeline.cli.app", "build"],
        ["python", "-m", "mlops_pipeline.cli.app", "prepare"],
        ["python", "-m", "mlops_pipeline.cli.app", "train-model"],
    ]

    for cmd in commands:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    logger.success("Full pipeline completed successfully")