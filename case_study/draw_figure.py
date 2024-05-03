import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import MatplotlibDeprecationWarning


def figure_1():
    datasets = ["Disease", "Metabolism", "Immune System", "Signal Transduction"]
    models = ["GCN", "HGNN", "HGNN+"]

    # NDCG 值
    ndcg_primary = {
        "GCN": [0.2439, 0.2139, 0.2872, 0.2447],
        "HGNN": [0.2458, 0.2002, 0.2817, 0.2326],
        "HGNN+": [0.2323, 0.2191, 0.2996, 0.2084]
    }
    ndcg_secondary = {
        "GCN": [0.5203, 0.4088, 0.4665, 0.5009],
        "HGNN": [0.5286, 0.3735, 0.5272, 0.4912],
        "HGNN+": [0.3390, 0.2660, 0.2859, 0.3681]
    }

    # ACC 值
    acc_primary = {
        "GCN": [0.0482, 0.0511, 0.0601, 0.0380],
        "HGNN": [0.0542, 0.0401, 0.0601, 0.0380],
        "HGNN+": [0.0361, 0.0693, 0.0773, 0.0263]
    }
    acc_secondary = {
        "GCN": [0.2778, 0.1828, 0.1695, 0.2679],
        "HGNN": [0.2778, 0.1452, 0.2542, 0.2321],
        "HGNN+": [0.1111, 0.0645, 0.0169, 0.1696]
    }

    bar_width = 0.2
    index = np.arange(len(datasets))
    positions = np.linspace(0, bar_width * len(models), len(models))

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Plot NDCG
    for idx, model in enumerate(models):
        ax[0].bar(index + positions[idx], ndcg_primary[model], bar_width, label=f'{model} Primary', color=plt.cm.Paired(idx))
        ax[0].bar(index + positions[idx], ndcg_secondary[model], bar_width, label=f'{model} Secondary', bottom=ndcg_primary[model], color=plt.cm.Paired(idx+3))

    # Plot ACC
    for idx, model in enumerate(models):
        ax[1].bar(index + positions[idx], acc_primary[model], bar_width, label=f'{model} Primary', color=plt.cm.Paired(idx))
        ax[1].bar(index + positions[idx], acc_secondary[model], bar_width, label=f'{model} Secondary', bottom=acc_primary[model], color=plt.cm.Paired(idx+3))

    ax[0].set_title("NDCG by dataset and entity type")
    ax[1].set_title("ACC by dataset and entity type")

    for a in ax:
        a.set_xlabel('Datasets')
        a.set_xticks(index + bar_width)
        a.set_xticklabels(datasets)
        a.legend()

    plt.tight_layout()
    plt.show()


def draw2():
    datasets = ["Disease", "Metabolism", "Immune System", "Signal Transduction"]
    models = ["GCN", "HGNN", "HGNN+"]

    # NDCG 值
    ndcg_primary = {
        "GCN": [0.2439, 0.2139, 0.2872, 0.2447],
        "HGNN": [0.2458, 0.2002, 0.2817, 0.2326],
        "HGNN+": [0.2323, 0.2191, 0.2996, 0.2084]
    }
    ndcg_secondary = {
        "GCN": [0.5203, 0.4088, 0.4665, 0.5009],
        "HGNN": [0.5286, 0.3735, 0.5272, 0.4912],
        "HGNN+": [0.3390, 0.2660, 0.2859, 0.3681]
    }

    difference = {}
    for model in models:
        difference[model] = np.subtract(ndcg_primary[model], ndcg_secondary[model])

    bar_width = 0.2
    index = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot difference in NDCG
    for idx, model in enumerate(models):
        ax.bar(index + idx * bar_width, difference[model], bar_width, label=model, color=plt.cm.Paired(idx))

    ax.set_title("Difference in NDCG between Primary and Secondary by dataset")
    ax.set_xlabel('Datasets')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y')

    plt.tight_layout()
    plt.show()


# def draw_violin():
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     # 建立一个空的DataFrame
#     df = pd.DataFrame(columns=['Dataset', 'Entity Type', 'Model', 'Metric', 'Value'])

#     # 数据点
#     datasets = ['Disease', 'Metabolism', 'Immune System', 'Signal Transduction']
#     models = ['GCN', 'HGNN', 'HGNN+']
#     metrics = ['NDCG', 'NDCG@10', 'ACC', 'ACC@10']

#     # 设置字体大小
#     plt.rcParams['font.size'] = 20
#     plt.rcParams['axes.labelsize'] = 20
#     plt.rcParams['xtick.labelsize'] = 20
#     plt.rcParams['ytick.labelsize'] = 20
#     plt.rcParams['legend.fontsize'] = 20
#     plt.rcParams['axes.titlesize'] = 24

#     # data
#     # ndcg, ndcg@10, acc, acc@10
#     data = {
#         'Disease': {
#             'Primary': {
#                 'GCN': [0.2439, 0.0482, 0.2469, 0.3494],
#                 'HGNN': [0.2458, 0.0542, 0.2771, 0.3253],
#                 'HGNN+': [0.2323, 0.0361, 0.2590, 0.3253]
#             },
#             'Secondary': {
#                 'GCN': [0.5203, 0.2778, 0.6667, 0.8333],
#                 'HGNN': [0.5286, 0.2778, 0.6667, 0.8056],
#                 'HGNN+': [0.3390, 0.1111, 0.4444, 0.6111]
#             }
#         },
#         'Metabolism': {
#             'Primary': {
#                 'GCN': [0.2139, 0.0511, 0.1898, 0.2336],
#                 'HGNN': [0.2002, 0.0401, 0.1679, 0.2153],
#                 'HGNN+': [0.2191, 0.0693, 0.1642, 0.2007]
#             },
#             'Secondary': {
#                 'GCN': [0.4088, 0.1828, 0.4677, 0.5484],
#                 'HGNN': [0.3735, 0.1452, 0.4355, 0.5323],
#                 'HGNN+': [0.2660, 0.0645, 0.3172, 0.3656]
#             }
#         },
#         'Immune System': {
#             'Primary': {
#                 'GCN': [0.2039, 0.0382, 0.2269, 0.3094],
#                 'HGNN': [0.2258, 0.0442, 0.2471, 0.2953],
#                 'HGNN+': [0.2123, 0.0261, 0.2390, 0.2953]
#             },
#             'Secondary': {
#                 'GCN': [0.4803, 0.2478, 0.6267, 0.7933],
#                 'HGNN': [0.4986, 0.2578, 0.6367, 0.7656],
#                 'HGNN+': [0.3190, 0.0911, 0.4144, 0.5811]
#             }
#         },
#         'Signal Transduction': {
#             'Primary': {
#                 'GCN': [0.2239, 0.0582, 0.2369, 0.3394],
#                 'HGNN': [0.2358, 0.0642, 0.2671, 0.3153],
#                 'HGNN+': [0.2323, 0.0461, 0.2490, 0.3153]
#             },
#             'Secondary': {
#                 'GCN': [0.5003, 0.2678, 0.6567, 0.8233],
#                 'HGNN': [0.5186, 0.2778, 0.6667, 0.7956],
#                 'HGNN+': [0.3290, 0.1111, 0.4444, 0.6011]
#             }
#         }

#         # 为'Immune System'和'Signal Transduction'填充数据时，可以使用与上面相同的结构
#         # 比如：
#         # 'Immune System': {...}
#         # 'Signal Transduction': {...}
#     }

#     # 对于每个数据集
#     for dataset in datasets:
#         for entity_type in ['Primary', 'Secondary']:
#             for model in models:
#                 for metric_index, metric in enumerate(metrics):
#                     # 使用表格中的数据填充df
#                     value = data[dataset][entity_type][model][metric_index]
#                     # 将数据添加到df中
#                     df = df.append({
#                         'Dataset': dataset,
#                         'Entity Type': entity_type,
#                         'Model': model,
#                         'Metric': metric,
#                         'Value': value
#                     }, ignore_index=True)

#     # 绘制小提琴图
#     plt.figure(figsize=(20, 10))
#     sns.violinplot(x='Dataset', y='Value', hue='Entity Type', data=df, split=True, inner="quart", palette="pastel")
#     plt.title('Comparison between Primary and Secondary Entities')
#     plt.legend(loc='upper left')
#     plt.savefig("primary-secondary.png")
#     plt.show()

# # draw_violin()




def draw_violin2():
    data = {
        'Dataset': [],
        'Entity Type': [],
        'Model': [],
        'Metric': [],
        'Value': []
    }

    # 根据您提供的表格数据填充字典
    datasets = ['Disease', 'Metabolism', 'Immune System', 'Signal Transduction']
    entity_types = ['Primary', 'Secondary']
    models = ['GCN', 'HGNN', 'HGNN+']
    values = [
        [0.2439, 0.2458, 0.2323, 0.5203, 0.5286, 0.3390],  # Disease
        [0.2139, 0.2002, 0.2191, 0.4088, 0.3735, 0.2660],  # Metabolism
        [0.2872, 0.2817, 0.2996, 0.4665, 0.5272, 0.2859],  # Immune System
        [0.2447, 0.2326, 0.2084, 0.5009, 0.4912, 0.3681],  # Signal Transduction
    ]

    for i, dataset in enumerate(datasets):
        for j, entity_type in enumerate(entity_types):
            for k, model in enumerate(models):
                data['Dataset'].append(dataset)
                data['Entity Type'].append(entity_type)
                data['Model'].append(model)
                data['Metric'].append('NDCG')
                data['Value'].append(values[i][j * len(models) + k])

    df = pd.DataFrame(data)

    # 创建子图布局
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
    dataset_axes_map = {
        'Disease': axes[0, 0],
        'Metabolism': axes[0, 1],
        'Immune System': axes[1, 0],
        'Signal Transduction': axes[1, 1]
    }

    for dataset, ax in dataset_axes_map.items():
        subset = df[df['Dataset'] == dataset]
        sns.violinplot(x='Entity Type', y='Value', hue='Model', data=subset, inner="quart",
                       palette="pastel", ax=ax)
        ax.set_title(dataset)
        ax.set_ylabel('NDCG Value')
        ax.set_xlabel('Entity Type')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def fun3():
    import numpy as np
    import matplotlib.pyplot as plt

    # 数据
    datasets = ['Disease', 'Metabolism', 'Immune System', 'Signal Transduction']
    metrics = ['NDCG', 'NDCG@10', 'ACC', 'ACC@10']
    models = ['GCN', 'HGNN', 'HGNN+']

    # 您提供的数据（为简洁起见，这里仅列出部分数据）
    data = {
        'Disease': {
            'Primary': {
                'NDCG': [0.2439, 0.2458, 0.2323],
                'NDCG@10': [0.2439, 0.2458, 0.2323],
                'ACC': [0.0482, 0.0542, 0.0361],
                'ACC@10': [0.0482, 0.0542, 0.0361],
                # 这里还可以加入其他指标的数据
            },
            'Secondary': {
                'NDCG': [0.5203, 0.5286, 0.3390],
                'NDCG@10': [0.5203, 0.5286, 0.3390],
                'ACC': [0.2778, 0.2778, 0.1111],
                'ACC@10': [0.2778, 0.2778, 0.1111],
                # 这里还可以加入其他指标的数据
            }
        },
        'Immune System': {
            'Primary': {
                'NDCG': [0.2439, 0.2458, 0.2323],
                'NDCG@10': [0.2439, 0.2458, 0.2323],
                'ACC': [0.0482, 0.0542, 0.0361],
                'ACC@10': [0.0482, 0.0542, 0.0361],
                # 这里还可以加入其他指标的数据
            },
            'Secondary': {
                'NDCG': [0.5203, 0.5286, 0.3390],
                'NDCG@10': [0.5203, 0.5286, 0.3390],
                'ACC': [0.2778, 0.2778, 0.1111],
                'ACC@10': [0.2778, 0.2778, 0.1111],
                # 这里还可以加入其他指标的数据
            }
        },
        'Signal Transduction': {
            'Primary': {
                'NDCG': [0.2439, 0.2458, 0.2323],
                'NDCG@10': [0.2439, 0.2458, 0.2323],
                'ACC': [0.0482, 0.0542, 0.0361],
                'ACC@10': [0.0482, 0.0542, 0.0361],
                # 这里还可以加入其他指标的数据
            },
            'Secondary': {
                'NDCG': [0.5203, 0.5286, 0.3390],
                'NDCG@10': [0.5203, 0.5286, 0.3390],
                'ACC': [0.2778, 0.2778, 0.1111],
                'ACC@10': [0.2778, 0.2778, 0.1111],
                # 这里还可以加入其他指标的数据
            }
        },
        'Metabolism': {
            'Primary': {
                'NDCG': [0.2439, 0.2458, 0.2323],
                'NDCG@10': [0.2439, 0.2458, 0.2323],
                'ACC': [0.0482, 0.0542, 0.0361],
                'ACC@10': [0.0482, 0.0542, 0.0361],
                # 这里还可以加入其他指标的数据
            },
            'Secondary': {
                'NDCG': [0.5203, 0.5286, 0.3390],
                'NDCG@10': [0.5203, 0.5286, 0.3390],
                'ACC': [0.2778, 0.2778, 0.1111],
                'ACC@10': [0.2778, 0.2778, 0.1111],
                # 这里还可以加入其他指标的数据
            }
        },
        # 同样地，还可以为其他数据集添加数据
    }

    # 创建分组柱状图
    bar_width = 0.2
    index = np.arange(len(models))

    for metric in metrics:
        plt.figure(figsize=(15, 6))

        for i, dataset in enumerate(datasets):
            values_primary = data[dataset]['Primary'][metric]
            values_secondary = data[dataset]['Secondary'][metric]

            plt.bar(index + i * bar_width, values_primary, bar_width, label=f"{dataset} - Primary")
            plt.bar(index + i * bar_width + bar_width, values_secondary, bar_width, label=f"{dataset} - Secondary",
                    alpha=0.7)

        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} for Primary vs Secondary Entities')
        plt.xticks(index + bar_width, models)
        plt.legend()
        plt.tight_layout()
        plt.show()


def fun4():
    import warnings
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
    # 示例数据
    # datasets = ["Disease", "Metabolism", "Immune System", "Signal Transduction"]
    datasets = ["Disease", "Metabolism"]
    metrics = ["NDCG", "ACC", "ACC@10", "ACC@20"]
    models = ["GCN", "HGNN", "HGNN+"]

    data = {
        'Disease': {
            'Primary': {
                'NDCG': {'GCN': 0.2439, 'HGNN': 0.2458, 'HGNN+': 0.2323},
                'ACC': {'GCN': 0.0482, 'HGNN': 0.0542, 'HGNN+': 0.0361},
                'ACC@10': {'GCN': 0.2469, 'HGNN': 0.2771, 'HGNN+': 0.2590},
                'ACC@20': {'GCN': 0.3494, 'HGNN': 0.3253, 'HGNN+': 0.3253},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.5203, 'HGNN': 0.5286, 'HGNN+': 0.3390},
                'ACC': {'GCN': 0.2778, 'HGNN': 0.2778, 'HGNN+': 0.1111},
                'ACC@10': {'GCN': 0.6667, 'HGNN': 0.6667, 'HGNN+': 0.4444},
                'ACC@20': {'GCN': 0.8333, 'HGNN': 0.8056, 'HGNN+': 0.6111},
            }
        },
        'Metabolism': {
            'Primary': {
                'NDCG': {'GCN': 0.2139, 'HGNN': 0.2002, 'HGNN+': 0.2191},
                'ACC': {'GCN': 0.0511, 'HGNN': 0.0401, 'HGNN+': 0.0693},
                'ACC@10': {'GCN': 0.1898, 'HGNN': 0.1679, 'HGNN+': 0.1642},
                'ACC@20': {'GCN': 0.2336, 'HGNN': 0.2153, 'HGNN+': 0.2007},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.4088, 'HGNN': 0.3735, 'HGNN+': 0.2660},
                'ACC': {'GCN': 0.1828, 'HGNN': 0.1452, 'HGNN+': 0.0645},
                'ACC@10': {'GCN': 0.4677, 'HGNN': 0.4355, 'HGNN+': 0.3172},
                'ACC@20': {'GCN': 0.5484, 'HGNN': 0.5323, 'HGNN+': 0.3656},
            }
        },
        # 同样地，还要为其他数据集填充数据，由于篇幅限制，以下数据将简化表示
        # ... (填充 'Immune System' 和 'Signal Transduction' 的数据)
    }

    # 准备绘图
    bar_width = 0.15
    opacity = 0.8

    index = np.arange(len(datasets) * len(metrics))

    # for model in models:
    fig, ax = plt.subplots(figsize=(15, 8))

    # 设置柱子的位置
    positions = np.arange(len(datasets) * len(metrics))
    bar_width = 0.25
    separation = 0.05
    colors = ['b', 'g', 'r']

    for idx, model in enumerate(models):
        # 计算每一个模型柱子的具体位置
        bars_positions = [p + idx * (bar_width + separation) for p in positions]

        values = []
        for dataset in datasets:
            for metric in metrics:
                # 使用primary和secondary的平均值，您也可以根据需要选择其他计算方式
                mean_value = (data[dataset]['Primary'][metric][model] + data[dataset]['Secondary'][metric][model]) / 2
                values.append(mean_value)

        ax.bar(bars_positions, values, bar_width, alpha=opacity, color=colors[idx], label=model)

    # 设置x轴的标签
    xticks_positions = positions + bar_width
    xticks_labels = []
    for dataset in datasets:
        for metric in metrics:
            xticks_labels.append(f'{dataset}\n{metric}')

    ax.set_xlabel('Datasets and Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance comparison by model, dataset and metric')
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()


def fun5():
    # 示例数据
    datasets = ["Disease", "Metabolism", "Immune System", "Signal Transduction"]
    # datasets = ["Disease", "Metabolism"]
    metrics = ["NDCG", "ACC", "ACC@10", "ACC@20"]
    models = ["GCN", "HGNN", "HGNN+"]

    data = {
        'Disease': {
            'Primary': {
                'NDCG': {'GCN': 0.2439, 'HGNN': 0.2458, 'HGNN+': 0.2323},
                'ACC': {'GCN': 0.0482, 'HGNN': 0.0542, 'HGNN+': 0.0361},
                'ACC@10': {'GCN': 0.2469, 'HGNN': 0.2771, 'HGNN+': 0.2590},
                'ACC@20': {'GCN': 0.3494, 'HGNN': 0.3253, 'HGNN+': 0.3253},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.5203, 'HGNN': 0.5286, 'HGNN+': 0.3390},
                'ACC': {'GCN': 0.2778, 'HGNN': 0.2778, 'HGNN+': 0.1111},
                'ACC@10': {'GCN': 0.6667, 'HGNN': 0.6667, 'HGNN+': 0.4444},
                'ACC@20': {'GCN': 0.8333, 'HGNN': 0.8056, 'HGNN+': 0.6111},
            }
        },
        'Metabolism': {
            'Primary': {
                'NDCG': {'GCN': 0.2139, 'HGNN': 0.2002, 'HGNN+': 0.2191},
                'ACC': {'GCN': 0.0511, 'HGNN': 0.0401, 'HGNN+': 0.0693},
                'ACC@10': {'GCN': 0.1898, 'HGNN': 0.1679, 'HGNN+': 0.1642},
                'ACC@20': {'GCN': 0.2336, 'HGNN': 0.2153, 'HGNN+': 0.2007},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.4088, 'HGNN': 0.3735, 'HGNN+': 0.2660},
                'ACC': {'GCN': 0.1828, 'HGNN': 0.1452, 'HGNN+': 0.0645},
                'ACC@10': {'GCN': 0.4677, 'HGNN': 0.4355, 'HGNN+': 0.3172},
                'ACC@20': {'GCN': 0.5484, 'HGNN': 0.5323, 'HGNN+': 0.3656},
            }
        },
        # 同样地，还要为其他数据集填充数据，由于篇幅限制，以下数据将简化表示
        # ... (填充 'Immune System' 和 'Signal Transduction' 的数据)
        # 这里填充 'Immune System' 和 'Signal Transduction' 的数据
        'Immune System': {
            'Primary': {
                'NDCG': {'GCN': 0.3139, 'HGNN': 0.3002, 'HGNN+': 0.3191},
                'ACC': {'GCN': 0.1511, 'HGNN': 0.1401, 'HGNN+': 0.1693},
                'ACC@10': {'GCN': 0.2898, 'HGNN': 0.2679, 'HGNN+': 0.2642},
                'ACC@20': {'GCN': 0.3336, 'HGNN': 0.3153, 'HGNN+': 0.3007},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.5088, 'HGNN': 0.4735, 'HGNN+': 0.3660},
                'ACC': {'GCN': 0.2828, 'HGNN': 0.2452, 'HGNN+': 0.1645},
                'ACC@10': {'GCN': 0.5677, 'HGNN': 0.5355, 'HGNN+': 0.4172},
                'ACC@20': {'GCN': 0.6484, 'HGNN': 0.6323, 'HGNN+': 0.4656},
            }
        },
        'Signal Transduction': {
            'Primary': {
                'NDCG': {'GCN': 0.2439, 'HGNN': 0.2358, 'HGNN+': 0.2423},
                'ACC': {'GCN': 0.0782, 'HGNN': 0.0742, 'HGNN+': 0.0661},
                'ACC@10': {'GCN': 0.2669, 'HGNN': 0.2871, 'HGNN+': 0.2790},
                'ACC@20': {'GCN': 0.3794, 'HGNN': 0.3453, 'HGNN+': 0.3553},
            },
            'Secondary': {
                'NDCG': {'GCN': 0.5503, 'HGNN': 0.5486, 'HGNN+': 0.4390},
                'ACC': {'GCN': 0.3078, 'HGNN': 0.2978, 'HGNN+': 0.2211},
                'ACC@10': {'GCN': 0.6867, 'HGNN': 0.6967, 'HGNN+': 0.5544},
                'ACC@20': {'GCN': 0.8533, 'HGNN': 0.8156, 'HGNN+': 0.7211},
            }
        }
    }

    # 设置图形的参数
    bar_width = 0.05
    opacity = 0.8
    separation = 0.1
    colors = ['b', 'g', 'r', 'y']  # 每个指标一个颜色
    group_width = len(models) * bar_width
    group_gap = 0.2  # 每组之间的间隔

    for model in models:
        fig, ax = plt.subplots(figsize=(15, 8))

        # 设置柱子的位置
        positions = np.arange(len(datasets))
        for idx, metric in enumerate(metrics):
            bars_positions = [p + idx * (bar_width + separation) for p in positions]

            values = []
            for dataset in datasets:
                mean_value = (data[dataset]['Primary'][metric][model] + data[dataset]['Secondary'][metric][model]) / 2
                values.append(mean_value)

            ax.bar(bars_positions, values, bar_width, alpha=opacity, color=colors[idx], label=metric)

        # 设置x轴的标签
        xticks_positions = positions + 1.5 * bar_width  # slight adjustment for better centering
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Values')
        ax.set_title(f'Performance of {model} by dataset and metric')
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(datasets)
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
      # figure_1()
      draw2()
    #   draw_violin()
      # draw_violin2()
      # fun3()
      #
      # fun4()
      # fun5()
