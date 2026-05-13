import pickle
from pathlib import Path
# 指向你的 pkl 路径
path = r"D:\energy_conservation\output\ato_phase_results_v3\梅堰-永茂路\multi_class_phase_artifacts.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)
print("当前 pkl 包含的等级基因:", data.keys())
if "class2" in data:
    print("Class 2 原始参考时间:", data["class2"]["time_ref_raw"])