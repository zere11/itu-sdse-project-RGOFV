package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(nil))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	repo := client.Host().Directory(".")

	ctr := client.Container().
		From("python:3.11").
		WithDirectory("/workspace", repo).
		WithWorkdir("/workspace").
		WithEnvVariable("PYTHONUNBUFFERED", "1")

	// -----------------------------------------------------
	// 1) Check requirements & install only if necessary
	// -----------------------------------------------------
	fmt.Println("Checking dependencies...")
	ctr = ctr.WithExec([]string{"python", "-m", "pip", "install", "--upgrade", "pip", "--quiet"})

	ctr = ctr.WithExec([]string{
		"sh", "-c",
		`python - <<'EOF'
import pkgutil
import os
missing=[]
# Try root requirements.txt first, then notebooks/requirements.txt
req_file = "requirements.txt"
if not os.path.exists(req_file):
    req_file = "notebooks/requirements.txt"
if not os.path.exists(req_file):
    print("WARNING: No requirements.txt found, skipping dependency check")
    exit(0)

reqs=open(req_file).read().splitlines()
for r in reqs:
    r = r.strip()
    if not r or r.startswith("#"):
        continue
    pkg=r.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
    if not pkgutil.find_loader(pkg):
        missing.append(r)
if missing:
    import subprocess
    print("Installing missing packages:", missing)
    subprocess.check_call(["pip","install"] + missing)
else:
    print("All Python requirements already installed.")
EOF`,
	})

	// -----------------------------------------------------
	// Helper to run a command, print stdout+stderr, fail on error
	// Dagger doesn't return stdout when commands fail, so we use a wrapper
	// -----------------------------------------------------
	logExec := func(stage string, c *dagger.Container, cmd []string) *dagger.Container {
		fmt.Println("\n=== " + stage + " ===")
		fmt.Printf("Running command: %v\n", cmd)

		// Build the actual command string
		var cmdStr string
		if len(cmd) >= 2 && cmd[0] == "sh" && cmd[1] == "-c" {
			cmdStr = cmd[2]
		} else {
			// Escape arguments for shell
			escaped := make([]string, len(cmd))
			for i, arg := range cmd {
				escaped[i] = "'" + strings.ReplaceAll(arg, "'", "'\"'\"'") + "'"
			}
			cmdStr = strings.Join(escaped, " ")
		}

		// Wrapper that always succeeds (exit 0) so Dagger returns stdout
		// We save the real exit code to a file and read it separately
		wrapper := fmt.Sprintf(`OUTPUT_FILE="/tmp/dagger_cmd_output.txt"; EXIT_FILE="/tmp/dagger_exit_code.txt"; (%s) > "$OUTPUT_FILE" 2>&1; echo $? > "$EXIT_FILE"; cat "$OUTPUT_FILE"; cat "$EXIT_FILE" >&2`, cmdStr)

		exec := c.WithExec([]string{"sh", "-c", wrapper})

		// Get output - wrapper always exits 0 so this should work
		output, err := exec.Stdout(ctx)
		if err != nil {
			output, _ = exec.Stderr(ctx)
			if output == "" {
				output = fmt.Sprintf("Could not capture output: %v", err)
			}
		}

		// Get exit code from stderr (we wrote it there)
		exitCodeStr, _ := exec.Stderr(ctx)
		var exitCode int
		if exitCodeStr != "" {
			// Extract the exit code (it's the last line)
			lines := strings.Split(strings.TrimSpace(exitCodeStr), "\n")
			if len(lines) > 0 {
				fmt.Sscanf(lines[len(lines)-1], "%d", &exitCode)
			}
		}

		// Remove exit code from output if it's there
		if strings.Contains(output, exitCodeStr) {
			output = strings.Replace(output, exitCodeStr, "", 1)
			output = strings.TrimSpace(output)
		}

		if output != "" {
			fmt.Println("--- OUTPUT ---")
			fmt.Println(output)
		}

		// Check exit code
		if exitCode != 0 {
			log.Fatalf("%s failed with exit code: %d\nOUTPUT: %s", stage, exitCode, output)
		}

		return exec
	}

	// -----------------------------------------------------
	// 2) mkdir structure
	// -----------------------------------------------------
	// Clean mlruns to avoid stale/corrupted local tracking data between runs
	ctr = logExec("Reset mlruns store", ctr, []string{"sh", "-c", "rm -rf mlruns && mkdir -p mlruns/.trash"})

	ctr = logExec("Creating dirs", ctr, []string{"sh", "-c", "mkdir -p artifacts notebooks/artifacts mlruns mlruns/.trash"})

	// -----------------------------------------------------
	// 3) Copy data if available
	// -----------------------------------------------------
	ctr = logExec("Copying data", ctr, []string{"sh", "-c", `
if [ -f notebooks/artifacts/training_data.csv ]; then
    echo "✓ Copying training_data.csv to notebooks/artifacts/raw_data.csv"
    cp notebooks/artifacts/training_data.csv notebooks/artifacts/raw_data.csv
elif [ -f notebooks/artifacts/raw_data.csv ]; then
    echo "✓ raw_data.csv already exists in notebooks/artifacts/"
else
    echo "✗ ERROR: No data file found!" && exit 1
fi

# Also copy to artifacts/ root for 01_data.py to find
if [ -f notebooks/artifacts/raw_data.csv ]; then
    cp notebooks/artifacts/raw_data.csv artifacts/raw_data.csv
    echo "✓ Copied raw_data.csv to artifacts/ root"
fi
`})

	// -----------------------------------------------------
	// 4) Run: notebooks/01_data.py
	// -----------------------------------------------------
	ctr = logExec("Running data stage", ctr, []string{"python", "-u", "notebooks/01_data.py"})

	// -----------------------------------------------------
	// 5) Run: notebooks/02_model.py
	// -----------------------------------------------------
	ctr = logExec("Running model training", ctr, []string{"python", "-u", "notebooks/02_model.py"})

	// -----------------------------------------------------
	// 6) Export artifacts
	// -----------------------------------------------------
	fmt.Println("\nExporting artifacts...")
	artifactsDir := ctr.Directory("/workspace/artifacts")
	_, err = artifactsDir.Export(ctx, "./artifacts")
	if err != nil {
		log.Fatalf("Failed to export artifacts: %v", err)
	}
	fmt.Println("✓ Exported artifacts to ./artifacts")

	mlrunsDir := ctr.Directory("/workspace/mlruns")
	_, err = mlrunsDir.Export(ctx, "./mlruns")
	if err != nil {
		log.Fatalf("Failed to export mlruns: %v", err)
	}
	fmt.Println("✓ Exported mlruns to ./mlruns")

	fmt.Println("\n✓ Pipeline completed successfully!")
}
