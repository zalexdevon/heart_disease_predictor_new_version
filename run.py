from classifier.Mylib.myfuncs import read_yaml, sub_param_for_yaml_file
from classifier.constants import *
from pathlib import Path
import sys
import os
import yaml

params = read_yaml(Path(PARAMS_FILE_PATH))

P = params.data_transformation.no
T = params.model_trainer.model_name
PE = params.model_evaluation.data_transformation_no
E = params.model_evaluation.model_name

replace_dict = {
    "${P}": P,
    "${T}": T,
    "${PE}": PE,
    "${E}": E,
}

sub_param_for_yaml_file("config_p.yaml", "config.yaml", replace_dict)
sub_param_for_yaml_file("dvc_p.yaml", "dvc.yaml", replace_dict)

# TODO: d
print("dvc.yaml:\n")
with open("dvc.yaml", "r", encoding="utf-8") as yaml_file:
    content = yaml.safe_load(yaml_file)
    print(content)
# d

stage_name = sys.argv[1]
os.system(f"dvc repro {stage_name}")
