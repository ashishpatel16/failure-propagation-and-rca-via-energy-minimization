import subprocess
import os
import shutil
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class D4JManager:
    def __init__(self, d4j_path: str, java_home: str):
        self.d4j_path = Path(d4j_path)
        self.java_home = Path(java_home)
        
    def run_command(self, args: list[str], cwd: str) -> str:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(self.java_home)
        env["PATH"] = f"{self.java_home}/bin:{env['PATH']}"
        
        if self.d4j_path.is_file():
            cmd = [str(self.d4j_path)] + args
        else:
            cmd = [f"{self.d4j_path}/framework/bin/defects4j"] + args
            
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise Exception(f"D4J Command Failed: {' '.join(cmd)}\nError: {result.stderr}")
        return result.stdout

    def checkout(self, project: str, bug_id: str, output_dir: str) -> None:
        """Executes: defects4j checkout -p <project> -v <bug_id>b -w <output_dir>"""
        self.run_command(["checkout", "-p", project, "-v", f"{bug_id}b", "-w", output_dir], cwd=".")

    def compile(self, project_dir: str) -> None:
        """Executes: defects4j compile"""
        self.run_command(["compile"], cwd=project_dir)

    def test(self, project_dir: str) -> str:
        """Executes: defects4j test"""
        return self.run_command(["test"], cwd=project_dir)

    def cobertura_coverage(self, project_dir: str) -> str:
        """Executes: defects4j coverage (Run standard Cobertura coverage analysis)"""
        return self.run_command(["coverage"], cwd=project_dir)

    def run_test_with_agent(self, project_dir: str, agent_jar_path: str, agent_args: str) -> str:
        """Executes: java -javaagent:<agent_jar>=<agent_args> -cp <cp> org.junit.runner.JUnitCore <test_class>"""
        props = self.get_properties(project_dir)
        cp = props["cp.test"]
        
        env = os.environ.copy()
        env["JAVA_HOME"] = str(self.java_home)
        
        # Base command for running JUnit with our agent
        # We also add the agent jar to the classpath to ensure its classes are visible
        cmd = [
            f"{self.java_home}/bin/java",
            f"-javaagent:{agent_jar_path}={agent_args}",
            "-cp", f"{agent_jar_path}:{cp}",
            "org.junit.runner.JUnitCore"
        ]
        
        # Get failing tests (trigger tests in D4J terms)
        failing_tests = self.run_command(["export", "-p", "tests.trigger"], cwd=project_dir).strip().splitlines()
        
        results = []
        for test in failing_tests:
            if not test.strip(): continue
            # D4J returns ClassName::MethodName, but JUnitCore needs ClassName
            test_class = test.split("::")[0]
            test_cmd = cmd + [test_class]
            print(f"Running test class with agent: {test_class}")
            res = subprocess.run(test_cmd, cwd=project_dir, capture_output=True, text=True, env=env)
            results.append(res.stdout + res.stderr)
            
        return "\n".join(results)

    def get_properties(self, project_dir: str) -> dict[str, str]:
        """Executes: defects4j export -p <property> (for various properties)"""
        properties = {}
        # Exporting common properties
        for prop in ["dir.src.classes", "dir.src.tests", "dir.bin.classes", "dir.bin.tests", "cp.compile", "cp.test"]:
            val = self.run_command(["export", "-p", prop], cwd=project_dir).strip()
            properties[prop] = val
            
        # Workaround for projects (like Jsoup) that include older JUnit jars in their lib/ dir.
        # Ensure the D4J-provided JUnit (usually 4.11/4.12) takes precedence.
        if "cp.test" in properties:
            cp_elements = properties["cp.test"].split(":")
            d4j_junit = None
            for el in cp_elements:
                if "framework/projects/lib/junit" in el:
                    d4j_junit = el
                    break
            if d4j_junit:
                cp_elements.remove(d4j_junit)
                cp_elements.insert(0, d4j_junit)
                properties["cp.test"] = ":".join(cp_elements)
                
        return properties

    def get_buggy_methods(self, project: str, bug_id: str, project_dir: str) -> list[str]:
        """Identify buggy methods by analyzing git diff between BUGGY and FIXED versions."""
        patched_dict = self.identify_patched_functions(project, int(bug_id), project_dir)
        buggy_methods = []
        for fqns in patched_dict.values():
            buggy_methods.extend(fqns)
        return list(set(buggy_methods))

    def _get_patch_path(self, project: str, bug_id: int) -> Path:
        """Locates the source patch for a bug within the Defects4J framework."""
        if self.d4j_path.is_file():
            # D4J_PATH is typically .../defects4j/framework/bin/defects4j
            framework_path = Path(self.d4j_path).parent.parent
        else:
            framework_path = Path(self.d4j_path) / "framework"
            
        patch_path = framework_path / "projects" / project / "patches" / f"{bug_id}.src.patch"
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found at {patch_path}")
        return patch_path

    def write_patch_to_file(self, project: str, bug_id: int, target_dir: str) -> Path:
        """Writes the bug's patch to a patch.txt file in the target directory."""
        patch_src = self._get_patch_path(project, bug_id)
        patch_dest = Path(target_dir) / "patch.txt"
        shutil.copy(patch_src, patch_dest)
        logger.info(f"Patch written to {patch_dest}")
        return patch_dest

    def identify_patched_functions(self, project: str, bug_id: int, target_dir: str) -> Dict[str, List[str]]:
        """
        Analyzes the patch and identifies the specific Java functions/methods
        to which the patch was applied.
        """
        patch_path = self._get_patch_path(project, bug_id)
        target_path = Path(target_dir).resolve()
        
        patched_data = {} # {file_path: [method_names]}
        
        current_file = None
        with open(patch_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Identify the file being modified
            if line.startswith("--- ") or line.startswith("+++ "):
                if "/dev/null" in line: continue
                # Extract relative path (assuming a/ or b/ prefix)
                parts = line.split()
                if len(parts) > 1:
                    path_str = parts[1]
                    if path_str.startswith("a/") or path_str.startswith("b/"):
                        path_str = path_str[2:]
                    current_file = path_str
                    if current_file not in patched_data:
                        patched_data[current_file] = set()

            # Identify the hunk start line
            elif line.startswith("@@"):
                if not current_file: continue
                
                # Format: @@ -start,len +start,len @@
                match = re.search(r"@@ -(\d+)", line)
                if match:
                    start_line = int(match.group(1))
                    try:
                        fqn = self._find_method_fqn(target_path / current_file, start_line)
                        patched_data[current_file].add(fqn)
                    except ValueError as e:
                        # Skip lines not inside a method (e.g., class-level fields, comments, imports)
                        logger.debug(f"Skipping patch line {start_line} in {current_file}: {e}")

        # Convert sets to sorted lists for output
        return {k: sorted(list(v)) for k, v in patched_data.items()}

    def _find_method_fqn(self, file_path: Path, line_num: int) -> str:
        """Heuristic to find the Fully Qualified Name (FQN) of the method containing a specific line number."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 1. Find the Package Name
        package_name = ""
        package_regex = re.compile(r'^\s*package\s+([\w\.]+)\s*;')
        for line in lines:
            match = package_regex.match(line)
            if match:
                package_name = match.group(1)
                break

        # 2. Find the Class Name (from filename)
        class_name = file_path.stem

        # 3. Find the Method Name (search backwards from line_num)
        method_name = None
        # Regex for common Java method signatures (avoiding 'return' statements)
        method_regex = re.compile(r'^\s*(?!(?:return|if|for|while|switch|catch|throw)\b)(?:(?:public|protected|private|static|final|native|synchronized|abstract|transient)\s+)*[\w\<\>\[\]]+\s+(\w+)\s*\([^\)]*\)\s*(?:throws\s+[\w\.\,\s]+)?\s*\{?')

        for i in range(min(line_num - 1, len(lines) - 1), -1, -1):
            line = lines[i]
            match = method_regex.search(line)
            if match:
                method_name = match.group(1)
                break
        
        if not method_name:
            raise ValueError(f"Could not find method name at line {line_num} in {file_path}")

        return f"{package_name}.{class_name}#{method_name}"

    def get_bug_ids(self, project: str) -> list[str]:
        """Executes: defects4j query -p <project> -q bug.id"""
        output = self.run_command(["query", "-p", project, "-q", "bug.id"], cwd=".")
        return [line.strip() for line in output.splitlines() if line.strip()]
