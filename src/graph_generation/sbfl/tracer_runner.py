import subprocess
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TracerConfig:
    tracer_dir: Path
    
    @property
    def src_dir(self) -> Path: return self.tracer_dir / "src"
    @property
    def bin_dir(self) -> Path: return self.tracer_dir / "bin"
    @property
    def lib_dir(self) -> Path: return self.tracer_dir / "lib"
    @property
    def manifest(self) -> Path: return self.tracer_dir / "MANIFEST.MF"
    @property
    def agent_jar(self) -> Path: return self.tracer_dir / "tracer-agent.jar"
    
    def validate(self) -> None:
        if not self.tracer_dir.exists():
            raise FileNotFoundError(f"Tracer directory missing: {self.tracer_dir}")
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory missing: {self.src_dir}")
        if not self.lib_dir.exists():
            raise FileNotFoundError(f"Lib directory missing: {self.lib_dir}")
        if not self.manifest.exists():
            raise FileNotFoundError(f"MANIFEST.MF missing: {self.manifest}")

class TracerRunner:
    def __init__(self, tracer_dir: str):
        self.config = TracerConfig(Path(tracer_dir))
        self.config.validate()
        
    def compile_agent(self) -> Path:
        if self.config.agent_jar.exists():
            return self.config.agent_jar
            
        logger.info(f"Compiling TracerAgent in {self.config.tracer_dir}")
        self.config.bin_dir.mkdir(exist_ok=True)
        
        java_files = list(self.config.src_dir.rglob("*.java"))
        if not java_files:
            raise ValueError(f"No .java files found in {self.config.src_dir}")
            
        java_file_paths = [str(p) for p in java_files]
        javassist_jar = self.config.lib_dir / "javassist.jar"
        
        if not javassist_jar.exists():
            raise FileNotFoundError(f"javassist.jar missing: {javassist_jar}")
        
        cmd_javac = ["javac", "-cp", str(javassist_jar), "-d", str(self.config.bin_dir)] + java_file_paths
        res = subprocess.run(cmd_javac, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Javac Compilation failed: {res.stderr}\n{res.stdout}")
            
        cmd_jar = ["jar", "cfm", str(self.config.agent_jar), str(self.config.manifest), "-C", str(self.config.bin_dir), "."]
        res_jar = subprocess.run(cmd_jar, capture_output=True, text=True)
        if res_jar.returncode != 0:
            raise RuntimeError(f"Jar packaging failed: {res_jar.stderr}\n{res_jar.stdout}")
            
        if not self.config.agent_jar.exists():
            raise RuntimeError(f"Compilation succeeded but {self.config.agent_jar} was not generated.")
            
        return self.config.agent_jar
