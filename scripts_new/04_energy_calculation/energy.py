import pandas as pd
import os
import glob
import warnings

# 忽略 openpyxl 样式警告
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# 1. 严格定义的 26 个站点顺序
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def calculate_ordered_metrics_in_seconds(date_folder_name):
    # 路径配置
    project_root = os.getcwd()
    target_folder = os.path.join(project_root, "data", "raw_data", date_folder_name)
    
    if not os.path.exists(target_folder):
        print(f"❌ 找不到文件夹: {target_folder}")
        return

    all_data_list = []
    print(f"🔍 正在严格按 26 站顺序处理文件夹: {date_folder_name} ...")

    # 2. 按顺序提取数据
    for section_name in LINE5_SECTIONS:
        search_pattern = os.path.join(target_folder, f"*{section_name}.xlsx")
        files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]
        
        if not files:
            # 如果某个区间文件不存在，打印提示
            # print(f"⚠️ 文件夹中缺少区间文件: {section_name}")
            continue
        
        file_path = files[0]
        try:
            # 读取“静态”工作表
            df = pd.read_excel(file_path, sheet_name='静态')
            df.columns = [str(c).strip() for c in df.columns] # 清洗表头空格

            # 动态匹配列名
            col_service = '服务号' if '服务号' in df.columns else None
            col_energy = next((c for c in df.columns if '能耗' in c and 'kwh' in c.lower()), None)
            col_time = next((c for c in df.columns if '实际' in c and '时间' in c), None)

            if col_service and col_energy and col_time:
                temp_df = df[[col_service, col_energy, col_time]].copy()
                temp_df.columns = ['service_id', 'energy', 'travel_time_s']
                temp_df['section'] = section_name
                all_data_list.append(temp_df)
            else:
                print(f"⚠️ 文件 {os.path.basename(file_path)} 表头匹配失败")

        except Exception as e:
            print(f"❌ 读取 {section_name} 出错: {e}")

    # 3. 数据分析与汇总
    if not all_data_list:
        print("未提取到任何有效数据。")
        return

    full_df = pd.concat(all_data_list, ignore_index=True)
    
    # 类型转换与清洗
    full_df['energy'] = pd.to_numeric(full_df['energy'], errors='coerce').fillna(0)
    full_df['travel_time_s'] = pd.to_numeric(full_df['travel_time_s'], errors='coerce').fillna(0)
    full_df['service_id'] = full_df['service_id'].astype(str)

    # 按服务号汇总
    summary = full_df.groupby('service_id').agg({
        'energy': 'sum',
        'travel_time_s': 'sum',
        'section': 'count'
    }).reset_index()

    # 重命名列名
    summary.columns = ['服务号', '全线总能耗(Kwh)', '实际全线运行时间(s)', '匹配成功区间数']
    
    # 4. 排序：按匹配数量从多到少（完整跑完26站的排在最前）
    summary = summary.sort_values(by=['匹配成功区间数', '服务号'], ascending=[False, True])

    # 5. 输出报表
    print("\n" + "="*80)
    print(f"📊 5号线全线汇总报表（秒级精度） | 日期目录: {date_folder_name}")
    print(f"说明：理想完整车次应包含 {len(LINE5_SECTIONS)} 个区间")
    print("="*80)
    
    # 打印格式化结果
    # 使用 round(2) 保留两位小数，看起来更整齐
    summary['全线总能耗(Kwh)'] = summary['全线总能耗(Kwh)'].round(2)
    summary['实际全线运行时间(s)'] = summary['实际全线运行时间(s)'].round(2)
    
    print(summary.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    # 执行汇总，只需更改这里的日期文件夹名
    calculate_ordered_metrics_in_seconds("05012-2024-07-08")