# -*- coding: utf-8 -*-
"""Generate e-s curve charts: Actual vs Physics (both from raw trip data)."""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.physics.train_simu import TrainTheoreticalEnergyModel

TRIP = 10
DATA_DIR = PROJECT_ROOT / 'data' / 'data_processed'
OUT_DIR = PROJECT_ROOT / f'output/energy_es_curves_trip{TRIP}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

engine = TrainTheoreticalEnergyModel()

for sp in STATIONS:
    raw_path = DATA_DIR / f'results_{sp}.xlsx'
    if not raw_path.exists():
        continue

    df_raw = pd.read_excel(raw_path)
    target_seg = sorted(df_raw['segment'].unique())[0]  # first trip

    df_seg = df_raw[df_raw['segment'] == target_seg].copy()

    # Actual velocity profile from raw data
    t_actual = df_seg['时刻'].values
    t_actual = t_actual - t_actual[0]
    v_actual = pd.to_numeric(df_seg['速度(m/s)'], errors='coerce').fillna(0).values
    s_actual = pd.to_numeric(df_seg['累计位移(m)'], errors='coerce').fillna(0).values
    s_actual = s_actual - s_actual[0]
    mass_ton = float(df_seg['重量'].iloc[0])

    # Physics energy — on actual velocity profile
    e_phy_steps = engine.run_batch_simulation(t_actual, v_actual, mass_ton)
    e_phy_cum = np.cumsum(e_phy_steps)
    e_phy_total = float(e_phy_cum[-1])

    # Actual measured energy
    e_actual_j = pd.to_numeric(df_seg['energy'], errors='coerce').fillna(0).values
    e_actual_wh = e_actual_j / 3600.0
    e_actual_cum = np.cumsum(e_actual_wh)
    e_act_total = float(e_actual_cum[-1])

    # Error
    phy_err_wh = e_phy_total - e_act_total
    phy_err_pct = phy_err_wh / e_act_total * 100 if e_act_total > 0 else 0

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(s_actual, e_actual_cum, color='#2874a6', linewidth=2.2,
            label=f'Actual (measured): {e_act_total:.0f} Wh')
    ax.plot(s_actual, e_phy_cum, color='#e74c3c', linewidth=2.0, linestyle='--',
            label=f'Physics (simulated): {e_phy_total:.0f} Wh | err={phy_err_wh:+.0f}Wh ({phy_err_pct:+.1f}%)')

    ax.set_title(f'{sp} | Trip{TRIP} seg0 | Energy-Distance | Physics vs Actual: {phy_err_wh:+.0f}Wh ({phy_err_pct:+.1f}%)')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Cumulative Energy (Wh)')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{sp}_energy_distance.png', dpi=250, bbox_inches='tight')
    plt.close()
    print(f'{sp}: actual={e_act_total:.0f}Wh  physics={e_phy_total:.0f}Wh  err={phy_err_wh:+.0f}Wh ({phy_err_pct:+.1f}%)')

print(f'\nDone -> {OUT_DIR}')
