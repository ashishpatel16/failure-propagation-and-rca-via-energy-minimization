import pandas as pd
import math
from typing import Dict

def map_line_to_method(line_name: str) -> str:
    """Maps a coverage column like 'com.example$App#process(int):30' to 'com.example.App#process(int)'."""
    if "#" not in line_name:
        return line_name.replace('$', '.')
    return line_name.split(':')[0].replace('$', '.')

def aggregate_coverage(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates line-level coverage into method-level coverage using OR logic.
    If any line in a method is covered (1), the method is considered covered (1).
    """
    method_data: Dict[str, pd.Series] = {}
    
    for col in coverage_df.columns:
        if col == 'Result':
            continue
        method: str = map_line_to_method(col)
        if method not in method_data:
            method_data[method] = coverage_df[col].copy()
        else:
            method_data[method] = method_data[method] | coverage_df[col]
            
    method_df: pd.DataFrame = pd.DataFrame(method_data)
    method_df['Result'] = coverage_df['Result']
    return method_df

def compute_tarantula(method_df: pd.DataFrame) -> Dict[str, float]:
    total_fail: int = len(method_df[method_df['Result'] == 'Fail'])
    total_pass: int = len(method_df[method_df['Result'] == 'Pass'])
    
    tarantula_scores: Dict[str, float] = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
            
        cf: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        if total_fail == 0 or (cf == 0 and cp == 0):
            tarantula: float = 0.0
        else:
            fail_ratio: float = cf / total_fail
            pass_ratio: float = cp / total_pass if total_pass > 0 else 0.0
            if fail_ratio + pass_ratio == 0:
                tarantula = 0.0
            else:
                tarantula = fail_ratio / (fail_ratio + pass_ratio)
                
        tarantula_scores[method] = float(tarantula)
        
    return tarantula_scores

def compute_ochiai(method_df: pd.DataFrame) -> Dict[str, float]:
    total_fail: int = len(method_df[method_df['Result'] == 'Fail'])
    
    ochiai_scores: Dict[str, float] = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
            
        cf: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        if total_fail == 0 or (cf + cp) == 0:
            ochiai: float = 0.0
        else:
            ochiai = cf / math.sqrt(total_fail * (cf + cp))
                
        ochiai_scores[method] = float(ochiai)
        
    return ochiai_scores

def compute_dstar(method_df: pd.DataFrame, star: int = 2) -> Dict[str, float]:
    total_fail: int = len(method_df[method_df['Result'] == 'Fail'])
    
    dstar_scores: Dict[str, float] = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
            
        cf: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp: int = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        nf: int = total_fail - cf
        denominator: int = cp + nf
        
        if denominator == 0:
            if cf > 0:
                dstar: float = float('inf')
            else:
                dstar = 0.0
        else:
            dstar = (cf ** star) / denominator
                
        dstar_scores[method] = float(dstar)
        
    return dstar_scores

