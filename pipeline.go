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

	// Base container (first stage): install requirements using ONLY requirements.txt
	base := client.Container().
		From("python:3.12.2-bookworm").
		WithMountedCache("/root/.cache/pip", pipCache).
		WithEnvVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1")

	// Copy requirements.txt and install; this layer will cache by the file content
	req := repo.File("requirements.txt")
	base = base.
		WithFile("/tmp/requirements.txt", req).
		WithExec([]string{
			"bash", "-lc",
			"python -m pip install --upgrade pip && python -m pip install -r /tmp/requirements.txt",
		})

	// Now mount the full repo and switch to /repo/notebooks
	base = base.
		WithDirectory("/repo", repo).
		WithWorkdir("/repo/notebooks").
		WithExec([]string{"python", "--version"})

	// Optional: quick ls for layout sanity (comment/remove once verified)
	// base = base.WithExec([]string{"bash", "-lc", "pwd && ls -al"})

	if _, err := base.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing data loading and preprocessing")
	dataPrep := base.WithExec([]string{"python", "01_data.py"})
	if _, err := dataPrep.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing training and registration")
	train := dataPrep.WithExec([]string{"python", "02_model.py"})
	if _, err := train.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Exporing Artifacts and Data")
	if _, err := train.Directory("/repo/notebooks/artifacts").Export(ctx, "artifacts"); err != nil {
		return err
	}
	if _, err := train.Directory("/repo/notebooks/mlruns").Export(ctx, "mlruns"); err != nil {
		return err
	}

	fmt.Println("Pipeline complete")
	return nil
}
