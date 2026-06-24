import os
import sys
import shutil

import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SBFLPregen")

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from graph_generation.sbfl.extractor import SBFLGraphExtractor
from visualization.graph_plots import plot_json_graph

PROJECTS = {
    # "Chart": "org.jfree",
    # "Cli": "org.apache.commons.cli",
    # "Closure": "com.google.javascript",
    # "Codec": "org.apache.commons.codec",
    # "Collections": "org.apache.commons.collections",
    # "Compress": "org.apache.commons.compress",
    # "Csv": "org.apache.commons.csv",
    # "Gson": "com.google.gson",
    # "JacksonCore": "com.fasterxml.jackson.core",
    # "JacksonDatabind": "com.fasterxml.jackson.databind",
    # "JacksonXml": "com.fasterxml.jackson.dataformat.xml",
    # "Jsoup": "org.jsoup",
    # "JxPath": "org.apache.commons.jxpath",
    # "Lang": "org.apache.commons.lang",
    # "Math": "org.apache.commons.math",
    "Mockito": "org.mockito",
    "Time": "org.joda.time"
}

TARGET_PROJECT = "Lang"
TARGET_BUG = "1"
RUN_ALL = True

def main() -> None:
    if not RUN_ALL and not TARGET_PROJECT:
        logger.error("Must specify either TARGET_PROJECT, or RUN_ALL")
        sys.exit(1)

    load_dotenv()
    
    JAVA_HOME = os.environ.get("JAVA_HOME_PATH")
    D4J_PATH = os.environ.get("D4J_PATH")
    
    if not JAVA_HOME or not Path(JAVA_HOME).exists():
        logger.error(f"JAVA_HOME_PATH invalid or missing: {JAVA_HOME}")
        sys.exit(1)
        
    if not D4J_PATH or not Path(D4J_PATH).exists():
        logger.error(f"D4J_PATH invalid or missing: {D4J_PATH}")
        sys.exit(1)
        
    extractor = SBFLGraphExtractor(d4j_path=D4J_PATH, java_home=JAVA_HOME)
    
    targets = []
    
    if RUN_ALL:
        for proj, prefix in PROJECTS.items():
            try:
                bugs = extractor.d4j_manager.get_bug_ids(proj)
                # if proj == "Jsoup":
                #     bugs = ['51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93']
                # print(bugs, proj)
                for b in bugs:
                    targets.append((proj, b, prefix))
            except Exception as e:
                logger.error(f"Failed to fetch bugs for project {proj}: {e}")
    else:
        if TARGET_PROJECT not in PROJECTS:
            logger.error(f"Unknown project {TARGET_PROJECT}. Available: {list(PROJECTS.keys())}")
            sys.exit(1)
        
        prefix = PROJECTS[TARGET_PROJECT]
        if TARGET_BUG:
            targets.append((TARGET_PROJECT, TARGET_BUG, prefix))
        else:
            try:
                bugs = extractor.d4j_manager.get_bug_ids(TARGET_PROJECT)
                for b in bugs:
                    targets.append((TARGET_PROJECT, b, prefix))
            except Exception as e:
                logger.error(f"Failed to fetch bugs for project {TARGET_PROJECT}: {e}")

    logger.info(f"Total targets to process: {len(targets)}")
    
    results = []
    
    for project, bug_id, package_prefix in targets:
        logger.info(f"Processing {project}-{bug_id}")
        output_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
        work_dir = output_dir / "workspace"
        
        try:
            extraction_res = extractor.extract(project, bug_id, str(output_dir), package_prefix)
            
            # Save buggy methods
            buggy_methods_path = output_dir / "buggy_methods.txt"
            with open(buggy_methods_path, "w") as f:
                for method in extraction_res.buggy_methods:
                    f.write(f"{method}\n")
            logger.info(f"Saved buggy methods to {buggy_methods_path}")
            
            # Copy original patch
            try:
                extractor.d4j_manager.write_patch_to_file(project, int(bug_id), str(output_dir))
            except Exception as patch_e:
                logger.warning(f"Failed to extract original patch for {project}-{bug_id}: {patch_e}")
            
            # Save visual plots
            try:
                plot_title = f"{project}-{bug_id} Call Graph"
                save_prefix = str(output_dir / "call_graph")
                plot_json_graph(extraction_res.graph_json, plot_title, save_prefix)
                logger.info(f"Saved call graph plots to {save_prefix}.png/svg")
            except Exception as plot_e:
                logger.warning(f"Failed to visualize graph: {plot_e}")
                
            results.append({
                "Project": project,
                "BugId": bug_id,
                "Status": "Success",
                "Total_Buggy_Methods": len(extraction_res.buggy_methods)
            })
            
        except Exception as e:
            logger.error(f"Failed to process {project}-{bug_id}: {e}", exc_info=True)
            results.append({
                "Project": project,
                "BugId": bug_id,
                "Status": f"Failed: {type(e).__name__} - {str(e)}",
                "Total_Buggy_Methods": 0
            })
        finally:
            # Clean up the workspace to save space
            if work_dir.exists():
                logger.info(f"Cleaning up workspace: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
                
            # Clean up the raw gzoltar output (including gzoltar.ser) to save space
            gzoltar_out_dir = output_dir / "gzoltar-out"
            if gzoltar_out_dir.exists():
                logger.info(f"Cleaning up raw gzoltar output: {gzoltar_out_dir}")
                shutil.rmtree(gzoltar_out_dir, ignore_errors=True)

    print("\n" + "="*80)
    print(f"{'BATCH EXTRACTION SUMMARY':^80}")
    print("="*80)
    if results:
        print(pd.DataFrame(results).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
