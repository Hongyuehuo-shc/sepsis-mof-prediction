"""Create the cross-database CSV tables and formatted Excel workbook."""
from __future__ import annotations
import os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node')
MODULES=Path(os.environ.get('NODE_PATH','')) if os.environ.get('NODE_PATH') else None
OUT=ROOT/'results/eicu_external_validation/cross_database_comparison'
subprocess.run([sys.executable,str(ROOT/'scripts/42_prepare_cross_database_tables.py'),'--output',str(OUT)],check=True)
if NODE and MODULES:
    env=os.environ.copy();env['NODE_PATH']=str(MODULES)
    subprocess.run([NODE,str(ROOT/'scripts/42_build_cross_database_workbook.mjs'),str(OUT)],check=True,env=env)
else:
    print('CSV comparison tables created. Optional Excel workbook export requires Node.js and @oai/artifact-tool.')
