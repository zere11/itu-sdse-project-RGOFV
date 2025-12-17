package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Host repository root
	repo := client.Host().Directory(".")

	// Persistent pip cache to speed up installs across runs
	pipCache := client.CacheVolume("pip-cache")

	// Base container (first stage): install requirements
	base := client.Container().
		From("python:3.12.2-bookworm").
		WithMountedCache("/root/.cache/pip", pipCache).
		WithEnvVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1").
		WithDirectory("/repo", repo).
		WithWorkdir("/repo/source").
		WithExec([]string{"python", "--version"})

	// Install requirements
	req := repo.File("requirements.txt")
	base = base.
		WithFile("/tmp/requirements.txt", req).
		WithExec([]string{
			"bash", "-lc",
			"python -m pip install --upgrade pip && python -m pip install -r /tmp/requirements.txt",
		})

	if _, err := base.Stdout(ctx); err != nil {
		return err
	}

	// Step 1: Dataset creation
	fmt.Println("Initializing dataset creation")
	dataset := base.WithExec([]string{"python", "makedataset.py"})
	if _, err := dataset.Stdout(ctx); err != nil {
		return err
	}

	// Step 2: Preprocessing
	fmt.Println("Initializing preprocessing")
	preprocess := dataset.WithExec([]string{"python", "preprocess.py"})
	if _, err := preprocess.Stdout(ctx); err != nil {
		return err
	}

	// Step 3: Feature engineering
	fmt.Println("Initializing feature engineering")
	features := preprocess.WithExec([]string{"python", "features.py"})
	if _, err := features.Stdout(ctx); err != nil {
		return err
	}

	// Step 4: Training
	fmt.Println("Initializing training and registration")
	train := features.WithExec([]string{"python", "train.py"})
	if _, err := train.Stdout(ctx); err != nil {
		return err
	}

	// Step 5: Export artifacts and mlflow
	fmt.Println("Exporting Artifacts and Data")
	if _, err := train.Directory("/repo/artifacts").Export(ctx, "artifacts"); err != nil {
		return err
	}
	if _, err := train.Directory("/repo/mlflow").Export(ctx, "mlflow"); err != nil {
		return err
	}

	fmt.Println("Pipeline complete")
	return nil
}
