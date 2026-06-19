import os
import sys
import shutil
import random
import logging
import tempfile
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from benchmarking.d4j_manager import D4JManager

def validate_bugs(manager: D4JManager, project: str, bug_ids: list[str]) -> list[dict]:
    results = []
    
    for bug_id in bug_ids:
        logging.info(f"Validating {project}-{bug_id}...")
        
        status = "Passed"
        extracted_methods = []
        modified_classes = []
        error_msg = ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. Checkout bug
                manager.checkout(project, bug_id, temp_dir)
                
                # 2. Extract buggy methods using our heuristic
                extracted_methods = manager.get_buggy_methods(project, bug_id, temp_dir)
                
                if not extracted_methods:
                    status = "Failed (No Methods Extracted)"
                else:
                    # 3. Get ground truth classes modified
                    classes_out = manager.run_command(["export", "-p", "classes.modified"], cwd=temp_dir)
                    modified_classes = [c.strip() for c in classes_out.splitlines() if c.strip()]
                    
                    if not modified_classes:
                        status = "Failed (No Modified Classes from D4J)"
                    else:
                        # 4. Verify class match
                        # extracted methods are like "org.apache.commons.lang3.StringUtils#join"
                        # We extract the class part
                        extracted_classes = set([m.split("#")[0] for m in extracted_methods])
                        
                        # We check if EVERY extracted class is at least a substring match to a modified class,
                        # or if there is an exact match. (Handling nested classes like A$B)
                        valid = True
                        for ec in extracted_classes:
                            # A simple check: does 'ec' match any modified class exactly, 
                            # or is it a prefix (e.g. nested classes)?
                            match_found = any(mc == ec or mc.startswith(ec + "$") or ec.startswith(mc + "$") for mc in modified_classes)
                            if not match_found:
                                valid = False
                                error_msg = f"Extracted class '{ec}' not in D4J modified classes {modified_classes}"
                                break
                                
                        if not valid:
                            status = f"Failed (Mismatch: {error_msg})"
                        
            except Exception as e:
                status = f"Failed (Exception: {type(e).__name__})"
                error_msg = str(e)
                logging.error(f"Error on {project}-{bug_id}: {e}", exc_info=True)
                
        results.append({
            "Project": project,
            "BugId": bug_id,
            "Status": status,
            "Methods_Found": len(extracted_methods),
            "Extracted_Methods": ", ".join(extracted_methods) if extracted_methods else "None",
            "Error": error_msg
        })
        
    return results

def main() -> None:
    load_dotenv()
    
    JAVA_HOME = os.environ.get("JAVA_HOME_PATH")
    D4J_PATH = os.environ.get("D4J_PATH")
    
    if not JAVA_HOME or not D4J_PATH:
        logging.error("JAVA_HOME_PATH or D4J_PATH not set in .env")
        return
        
    manager = D4JManager(d4j_path=D4J_PATH, java_home=JAVA_HOME)
    
    projects_to_test = ["Lang", "Compress"]
    samples_per_project = 5
    
    all_results = []
    
    for project in projects_to_test:
        all_bugs = manager.get_bug_ids(project)
        if not all_bugs:
            logging.warning(f"No bugs found for {project}")
            continue
            
        # For reproducibility, pick first 5 or shuffle with fixed seed
        random.seed(42)
        sample_bugs = random.sample(all_bugs, min(samples_per_project, len(all_bugs)))
        
        logging.info(f"Testing {project}: {sample_bugs}")
        res = validate_bugs(manager, project, sample_bugs)
        all_results.extend(res)
        
    print("\n" + "="*120)
    print(f"{'BUGGY METHOD EXTRACTION VALIDATION SUMMARY':^120}")
    print("="*120)
    df = pd.DataFrame(all_results)
    
    pd.set_option('display.max_colwidth', 50)
    print(df[["Project", "BugId", "Status", "Methods_Found", "Extracted_Methods"]].to_string(index=False))
    print("="*120)

if __name__ == "__main__":
    main()
