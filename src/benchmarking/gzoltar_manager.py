import os
import subprocess
import pandas as pd
from pathlib import Path

class GZoltarRunner:
    def __init__(self, gzoltar_dir: str, java_home: str):
        self.gzoltar_dir = Path(gzoltar_dir)
        self.cli_jar = self.gzoltar_dir / "gzoltarcli.jar"
        self.agent_jar = self.gzoltar_dir / "gzoltaragent.jar"
        self.java_home = Path(java_home)
        self.java_bin = self.java_home / "bin" / "java"

    def list_tests(self, test_classes_dir: str, cp: str, output_file: str) -> None:
        """Use GZoltar to list all unit tests in the project."""
        cmd = [
            str(self.java_bin),
            "-cp", f"{cp}:{self.cli_jar}",
            "com.gzoltar.cli.Main", "listTestMethods",
            test_classes_dir,
            "--outputFile", output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"GZoltar listTestMethods failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    def collect_coverage(self, project_dir: str, src_classes_dir: str, cp: str, unit_tests_file: str, output_dir: str, includes: str, excludes: str) -> str:
        """Run GZoltar collection on the specified project and tests."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Run tests with agent
        dest_file = Path(output_dir) / "gzoltar.ser"
        if dest_file.exists():
            dest_file.unlink() # Delete corrupted/old session data
            
        cmd = [
            str(self.java_bin),
            f"-javaagent:{self.agent_jar}=destfile={dest_file},buildlocation={src_classes_dir},includes={includes},excludes={excludes}",
            "-cp", f"{cp}:{self.cli_jar}",
            "com.gzoltar.cli.Main", "runTestMethods",
            "--testMethods", unit_tests_file,
            "--collectCoverage"
        ]
        
        print("Running tests with GZoltar agent...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_dir)
        if result.returncode != 0 and "Tests run:" not in result.stdout:
            raise Exception(f"GZoltar Collection Failed:\n{result.stderr}\n{result.stdout}")

        # 2. Generate Report (Matrix/Spectra)
        print("Generating GZoltar report (matrix/spectra)...")
        self.fault_localization_report(src_classes_dir, str(dest_file), output_dir, includes, excludes)
        return result.stdout

    def fault_localization_report(self, src_classes_dir: str, data_file: str, output_dir: str, includes: str, excludes: str) -> None:
        """Convert *.ser file to matrix/spectra txt files."""
        cmd = [
            str(self.java_bin),
            "-cp", f"{self.cli_jar}",
            "com.gzoltar.cli.Main", "faultLocalizationReport",
            "--buildLocation", src_classes_dir,
            "--dataFile", data_file,
            "--outputDirectory", output_dir,
            "--formatter", "TXT",
            "--includes", includes,
            "--excludes", excludes
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"GZoltar faultLocalizationReport failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    def export_to_csv(self, gzoltar_output_dir: str, output_csv_path: str) -> pd.DataFrame:
        """
        Critically parse the matrix and spectra files and merge them into a single CSV.
        """
        # GZoltar 1.7.x puts files in sfl/txt/
        base_path = Path(gzoltar_output_dir) / "sfl" / "txt"
        spectra_path = base_path / "spectra.csv"
        matrix_path = base_path / "matrix.txt"

        if not spectra_path.exists() or not matrix_path.exists():
            # Fallback for other versions
            spectra_path = Path(gzoltar_output_dir) / "spectra.txt"
            matrix_path = Path(gzoltar_output_dir) / "matrix.txt"
            if not spectra_path.exists() or not matrix_path.exists():
                raise FileNotFoundError(f"GZoltar output files not found in {gzoltar_output_dir} or {base_path}")

        # 1. Parse Spectra (Components)
        # spectra.csv has a header 'name', but signatures contain commas!
        with open(spectra_path, 'r') as f:
            lines = f.readlines()
            # The first line is 'name', the rest are components
            components = [line.strip() for line in lines[1:] if line.strip()]

        # 2. Parse Matrix (Tests x Components)
        rows = []
        with open(matrix_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts: continue
                
                outcome = parts[-1]
                coverage = [int(x) for x in parts[:-1]]
                
                if len(coverage) != len(components):
                    # Some versions might have extra bits at the end or start
                    # Let's be flexible but warned
                    coverage = coverage[:len(components)]
                
                row = {comp: cov for comp, cov in zip(components, coverage)}
                row['Result'] = 'Pass' if outcome == '+' else 'Fail'
                rows.append(row)

        if not rows:
            raise ValueError("The coverage matrix is empty. No tests were executed or recorded.")

        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c != 'Result'] + ['Result']
        df = df[cols]
        
        df.to_csv(output_csv_path, index=False)
        return df

if __name__ == "__main__":
    pass
