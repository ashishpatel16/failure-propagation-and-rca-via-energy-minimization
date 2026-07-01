import os
import json
import pytest
import csv
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = Path(os.getcwd())

deprecated_bugs_cache = {}

def is_deprecated(project: str, bug_id: str) -> bool:
    if project not in deprecated_bugs_cache:
        deprecated_bugs_cache[project] = set()
        csv_path = ROOT_DIR / "defects4j" / "framework" / "projects" / project / "deprecated-bugs.csv"
        if csv_path.exists():
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "bug.id" in row:
                        deprecated_bugs_cache[project].add(row["bug.id"])
    return bug_id in deprecated_bugs_cache[project]

def get_test_cases():
    """
    Finds all project_bug_id directories within CALLGRAPH_DIR.
    Returns a list of tuples: (project_bug_id, bug_dir_path) 
    """
    callgraph_dir = os.environ.get("CALLGRAPH_DIR")
    if not callgraph_dir:
        return []
        
    base_path = Path(callgraph_dir)
    if not base_path.exists() or not base_path.is_dir():
        return []
        
    cases = []
    csv_results = []
    
    for project_bug_dir in base_path.iterdir():
        if not project_bug_dir.is_dir():
            continue
            
        parts = project_bug_dir.name.rsplit("_", 1)
        if len(parts) == 2:
            project, bug_id = parts
            if is_deprecated(project, bug_id):
                csv_results.append({"project": project, "bug_id": bug_id, "status": "deprecated"})
                continue
                
            # Determine pass/fail status for the CSV
            status = "failed"
            callgraph_file = project_bug_dir / "call_graph.json"
            if callgraph_file.exists() and callgraph_file.is_file():
                try:
                    with open(callgraph_file, "r") as f:
                        data = json.load(f)
                    if len(data.get("nodes", [])) > 0:
                        status = "passed"
                except Exception:
                    pass
            
            csv_results.append({"project": project, "bug_id": bug_id, "status": status})
        cases.append((project_bug_dir.name, project_bug_dir))
        
    # Ensure all deprecated bugs for encountered projects are included in the CSV
    # even if they don't have a folder in CALLGRAPH_DIR
    existing_combinations = {(r["project"], r["bug_id"]) for r in csv_results}
    for project, bugs in deprecated_bugs_cache.items():
        for bug_id in bugs:
            if (project, bug_id) not in existing_combinations:
                csv_results.append({"project": project, "bug_id": bug_id, "status": "deprecated"})
        
    # Write out the CSV file
    csv_results.sort(key=lambda x: (x["project"], int(x["bug_id"]) if x["bug_id"].isdigit() else x["bug_id"]))
    with open(ROOT_DIR / "bug_status.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["project", "bug_id", "status"])
        writer.writeheader()
        writer.writerows(csv_results)
        
    return cases

test_cases = get_test_cases()

# If no cases are found, provide a dummy to allow pytest to collect the test 
# and explicitly skip it later rather than completely failing collection.
if not test_cases:
    test_cases = [("dummy_project", Path("dummy_dir"))]

@pytest.fixture(params=test_cases, ids=lambda x: f"{x[0]}")
def bug_case(request):
    """Fixture that parameterizes the test over all discovered bug directories."""
    return request.param

def test_callgraph_exists_and_has_nodes(bug_case):
    project_bug_id, bug_dir = bug_case
    
    # If the env var is not set, skip the test gracefully.
    if project_bug_id == "dummy_project" and not os.environ.get("CALLGRAPH_DIR"):
        pytest.skip("CALLGRAPH_DIR environment variable is not set or directory is empty.")
        
    callgraph_file = bug_dir / "call_graph.json"
    
    # Assert the call_graph.json file exists
    assert callgraph_file.exists(), f"call_graph.json is absent for {project_bug_id} at {callgraph_file}"
    assert callgraph_file.is_file(), f"{callgraph_file} is not a valid file"
    
    # Read the JSON and assert nodes are present
    with open(callgraph_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON format in {callgraph_file}")
            
    nodes = data.get("nodes", [])
    assert len(nodes) > 0, f"Number of nodes is 0 for {project_bug_id}"
