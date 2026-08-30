
import pandas as pd

DATA = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'
DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'

def crop_sales_2023():

    det = pd.read_csv(f'{DATA}/经济参数明细.csv', skipinitialspace=True)
    det['地块类型'] = det['地块类型'].astype(str).str.strip()
    det['种植季次'] = det['种植季次'].astype(str).str.strip()
    det['作物编号'] = pd.to_numeric(det['作物编号']).astype(int)
    det['亩产量/斤'] = pd.to_numeric(det['亩产量/斤'])
    yield_lookup = {}
    for _, r in det.iterrows():
        yield_lookup[(r['地块类型'], r['种植季次'], r['作物编号'])] = r['亩产量/斤']

    plots = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地')
    plot_type = dict(zip(plots['地块名称'].astype(str).str.strip(),
                         plots['地块类型'].astype(str).str.strip()))

    plant = pd.read_csv(f'{DATA}/2023种植情况.csv', skipinitialspace=True)
    plant['种植地块'] = plant['种植地块'].astype(str).str.strip()
    plant['种植季次'] = plant['种植季次'].astype(str).str.strip()
    plant['种植面积/亩'] = pd.to_numeric(plant['种植面积/亩'])

    sales = {}
    for _, r in plant.iterrows():
        j = int(r['作物编号'])
        k = (plot_type[r['种植地块']], r['种植季次'], j)
        sales[j] = sales.get(j, 0.0) + float(r['种植面积/亩']) * yield_lookup[k]
    return {j: round(v, 1) for j, v in sales.items()}

if __name__ == '__main__':
    s = crop_sales_2023()
    print(f'作物数={len(s)}, 总产量={sum(s.values()):.1f} 斤')
    for j in range(1, 42):
        print(f'  作物{j}: {s[j]:.0f} 斤')
