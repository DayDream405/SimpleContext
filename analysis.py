import json
from collections import Counter
import bisect
from typing import List, Tuple, Dict, Union, Optional
def main():
    # # 1. 构建或加载训练集索引
    # try:
    #     train_index = load_index("datasets/SemCor/train_freq_index.json")
    #     print("已加载预构建的训练集频率索引")
    # except FileNotFoundError:
    #     print("构建训练集频率索引...")
    #     path = 'datasets/SemCor/semcor_adjust.json'
    #     with open(path, 'r', encoding='utf-8') as f:
    #         train_data = json.load(f)
    #     train_index = build_frequency_index(train_data)
    #     save_index(train_index, "datasets/SemCor/train_freq_index_adjust.json")
    #     print("训练集频率索引已保存到 datasets/SemCor/train_freq_index_adjust.json")

    # path = 'saved_models/best_model_original/semeval2013_results.json'
    # with open(path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # # print(get_top_n_support(data, n=5))
    
    # counts = train_index["counts"]
    # categories = train_index["categories"]
    # counter = Counter(dict(zip(categories, counts)))
    # count = 0
    # class_num = 0
    # for cat, c in counter.items():
    #     if cat in data['class_wise_accuracy']:
    #         if data['class_wise_accuracy'][cat]['support'] > 0:
    #             class_num += 1
    #             if 0 < c <= 5:
    #                 count += 1
    # print(f"训练集中出现次数在1到5次的类别数量: {count}")
    # print(f"训练集中类别总数: {class_num}")

    # path = 'datasets/en/semeval2013.json'
    # with open(path, 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)

    # count = count_in_train(test_data,count_top_categories(test_data, n=10), return_rank=True, precomputed_index=train_index)
    # print(count)
    # print(get_train_top_n(train_data, n=5))
    # print(query_test_performance(data, get_train_top_n(train_index=train_index, n=100), test_has_accuracy=True))
    # print(calculate_single_occurrence_accuracy(None, test_data=data, train_index=train_index)['average_accuracy'])
    # print(calculate_high_accuracy_classes_stats(test_data=data, train_index=train_index, accuracy_min=0, accuracy_max=0.5, min_test_count=2)['average_train_count'])

    # word_level_statistics("saved_models/trimed_frequency5_drop_plus5_target_unified_20", "all", 'saved_models/best_model_original_unified_10/all_trimed_frequency5_drop_plus5_target_semcor.json')
    
    # len = average_context_length('/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', filter_list_path='bert_same_accuracy1.json')
    # print(f"Average context length: {len}")

    clb_results = get_clb('/mnt/zly/SpecializedTtraining/saved_models/best_model_original_unified_10/clb_results.json', 'eval2007_sense_accuracy.json')
    # print(clb_results)
    clbs = [v['CLB'] for v in clb_results.values()]
    print(f"Average CLB: {sum(clbs) / len(clbs) if clbs else 0}")
    print(f'Max CLB: {max(clbs) if clbs else 0}, Min CLB: {min(clbs) if clbs else 0}')

def get_clb(clb_results_path, filter_list_path):
    with open(clb_results_path, 'r', encoding='utf-8') as f:
        clb_results = json.load(f)
    with open(filter_list_path, 'r', encoding='utf-8') as f:
        filter_list = json.load(f)['sense_accuracy']
    with open('/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    key_map = {}
    for item in test_data:
        name = item['polysemous']['name']
        for definition in item['correct_definitions_in_context']:
            key = f"{name}::{definition}"
            key_map[key] = item['correct_definition_key'][0]
    filtered_clb = {k: v for k, v in clb_results.items() if key_map.get(k) in filter_list and filter_list[key_map[k]] == 1}
    return filtered_clb

def average_context_length(data_path, filter_list_path=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if filter_list_path:
        with open(filter_list_path, 'r', encoding='utf-8') as f:
            filter_list = set(json.load(f))
    else:
        filter_list = set()
    
    total_length = 0
    count = 0
    
    for item in data:
        keys = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        if filter_list_path and not any(key in filter_list for key in keys):
            continue
        
        context = item.get("context", "")
        total_length += len(context)
        count += 1
    
    average_length = total_length / count if count > 0 else 0
    return average_length

def get_top_n_support(data: Dict, n: int = 5) -> List[Tuple[str, int]]:
    """
    获取class_wise_accuracy中support最高的前n个类别
    
    参数:
        data: 包含class_wise_accuracy字段的字典
        n: 返回的top数量
    
    返回:
        [(类别名称, support值)] 列表，按support降序排列
    """
    if not isinstance(data, dict) or "class_wise_accuracy" not in data:
        return []
    
    # 提取并过滤support>0的类别
    classes = [
        (cls, metrics["support"])
        for cls, metrics in data["class_wise_accuracy"].items()
        if metrics.get("support", 0) > 0
    ]
    
    # 按support降序排序并返回前n个
    return sorted(classes, key=lambda x: x[1], reverse=True)[:n]

def count_top_categories(data: List[Dict], n: int = 5) -> List[Tuple[str, int]]:
    """
    统计数据集中出现频率最高的前n个类别
    
    参数:
        data: 原始数据集列表
        n: 返回的top数量
    
    返回:
        [(类别名称::定义, 出现次数)] 列表，按次数降序排列
    """
    if not isinstance(data, list):
        return []
    
    counter = Counter()
    for item in data:
        name = item.get("polysemous", {}).get("name", "")
        definitions = item.get("correct_definitions_in_context", [])
        
        for definition in definitions:
            category = f"{name}::{definition}"
            counter[category] += 1
    
    return counter.most_common(n)

def count_in_train(
    train_data: List[Dict],
    query_categories: Union[str, List[str], List[Tuple[str, int]]],
    return_rank: bool = False,
    precomputed_index: Dict = None
) -> Union[Dict[str, int], Dict[str, Tuple[int, int]]]:
    """
    查询一个或多个类别在训练集中的出现次数和排名
    
    参数:
        train_data: 训练数据集
        query_categories: 要查询的类别(可以是前两个函数的输出)
        return_rank: 是否返回排名
        precomputed_index: 预计算索引(可选)
    
    返回:
        如果return_rank=False: {类别: 次数}
        如果return_rank=True: {类别: (次数, 排名)}
    """
    # 统一处理输入格式
    if isinstance(query_categories, str):
        categories = [query_categories]
    elif isinstance(query_categories, list) and query_categories:
        if isinstance(query_categories[0], tuple):  # 前两个函数的输出格式
            categories = [cat for cat, _ in query_categories]
        else:
            categories = query_categories
    else:
        return {}
    
    # 使用预计算索引或实时计算
    if precomputed_index:
        counts = {cat: count for cat, count in zip(precomputed_index["categories"], precomputed_index["counts"])}
        rank_dict = precomputed_index["index"]
    else:
        # 实时计算训练集统计
        counts = Counter()
        for item in train_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            for definition in definitions:
                category = f"{name}::{definition}"
                counts[category] += 1
        
        # 创建排名字典
        sorted_categories = [cat for cat, _ in counts.most_common()]
        rank_dict = {cat: i+1 for i, cat in enumerate(sorted_categories)}
    
    # 构建结果
    result = {}
    for category in categories:
        count = counts.get(category, 0)
        if return_rank:
            rank = rank_dict.get(category, None)
            result[category] = (count, rank)
        else:
            result[category] = count
    
    return result

def get_train_top_n(
    train_data: Optional[List[Dict]] = None,
    train_index: Optional[Dict] = None,
    n: int = 5,
    min_count: int = 1
) -> List[Tuple[str, int]]:
    """
    获取训练集中出现频率最高的前n个类别
    
    参数:
        train_data: 训练数据集(与train_index二选一)
        train_index: 预计算索引(与train_data二选一)
        n: 返回的top数量
        min_count: 最小出现次数阈值
    
    返回:
        [(类别名称::定义, 出现次数)] 列表，按次数降序排列
    """
    # 验证输入
    if train_index is None and train_data is None:
        raise ValueError("必须提供train_data或train_index")
    
    # 获取计数器
    if train_index is not None:
        counter = train_index["counts"]
        sorted_categories = train_index["categories"]
    else:
        counter = Counter()
        for item in train_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            for definition in definitions:
                category = f"{name}::{definition}"
                counter[category] += 1
        sorted_categories = [cat for cat, _ in counter.most_common()]
    
    # 过滤并返回结果
    results = []
    for i, category in enumerate(sorted_categories):
        count = counter[i]
        if count >= min_count:
            results.append((category, count))
            if len(results) >= n:
                break
    
    return results

def query_test_performance(
    test_data: Union[Dict, List[Dict]], 
    train_top_categories: List[Tuple[str, int]],
    test_has_accuracy: bool = False,
    filter_zero_test: bool = True
) -> List[Dict[str, Union[str, int, float, None]]]:
    """
    查询训练集高频类别在测试集中的表现
    
    参数:
        test_data: 测试集数据(可以是原始数据或class_wise_accuracy格式)
        train_top_categories: get_train_top_n的输出结果
        test_has_accuracy: 测试数据是否包含准确率信息
    
    返回:
        包含以下字段的字典列表:
        - category: 类别名称
        - train_count: 训练集出现次数
        - test_count: 测试集出现次数(若无则为0)
        - test_rank: 测试集频率排名(若无则为None)
        - accuracy: 测试集准确率(若有且可获取)
    """
    # 准备测试集统计
    test_counter = Counter()
    test_accuracy = {}
    
    if isinstance(test_data, dict) and "class_wise_accuracy" in test_data:
        # 处理class_wise_accuracy格式
        for category, metrics in test_data["class_wise_accuracy"].items():
            test_counter[category] = metrics.get("support", 0)
            if test_has_accuracy:
                test_accuracy[category] = metrics.get("accuracy")
    elif isinstance(test_data, list):
        # 处理原始数据格式
        for item in test_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            for definition in definitions:
                category = f"{name}::{definition}"
                test_counter[category] += 1
    
    # 计算测试集排名
    test_rank_mapping = {
        cat: rank+1 
        for rank, (cat, _) in enumerate(test_counter.most_common())
    }
    
    # 查询每个训练集高频类别在测试集中的表现
    results = []
    for category, train_count in train_top_categories:
        test_count = test_counter.get(category, 0)
        if filter_zero_test and test_count == 0:
            continue
        test_rank = test_rank_mapping.get(category)
        
        result = {
            "category": category,
            "train_count": train_count,
            "test_count": test_count,
            "test_rank": test_rank,
        }
        
        if test_has_accuracy:
            result["accuracy"] = test_accuracy.get(category)
        
        results.append(result)
    
    return results

def calculate_single_occurrence_accuracy(
    train_data: List[Dict],
    test_data: Union[Dict, List[Dict]],
    train_index: Optional[Dict] = None
) -> Dict[str, Union[float, int, List[Tuple[str, float]]]]:
    """
    统计训练集中只出现一次的类别在测试集中的平均准确率
    
    参数:
        train_data: 训练数据集
        test_data: 测试集数据
        train_index: 预计算训练集索引(可选)
    
    返回:
        {
            "average_accuracy": 平均准确率,
            "total_count": 符合条件的类别总数,
            "test_matched_count": 在测试集中有记录的类别数,
            "details": [(类别, 准确率)]  # 测试集中有记录的类别详情
        }
    """
    # 1. 获取训练集中只出现一次的类别
    if train_index is None:
        train_counter = Counter()
        for item in train_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            for definition in definitions:
                category = f"{name}::{definition}"
                train_counter[category] += 1
    else:
        train_counter = train_index["counts"]
        category = train_index["categories"]
    
    single_categories = [cat for cat, count in zip(category, train_counter) if count == 1]
    
    # 2. 从测试集中提取这些类别的准确率
    test_accuracy = {}
    
    if isinstance(test_data, dict) and "class_wise_accuracy" in test_data:
        # 处理class_wise_accuracy格式
        for category, metrics in test_data["class_wise_accuracy"].items():
            if category in single_categories and "accuracy" in metrics:
                test_accuracy[category] = metrics["accuracy"]
    elif isinstance(test_data, list):
        # 处理原始数据格式(假设需要计算准确率)
        correct_counts = Counter()
        total_counts = Counter()
        
        for item in test_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            predicted = item.get("predicted_definition", "")
            
            for definition in definitions:
                category = f"{name}::{definition}"
                if category in single_categories:
                    total_counts[category] += 1
                    if definition == predicted:
                        correct_counts[category] += 1
        
        # 计算每个类别的准确率
        for category in total_counts:
            test_accuracy[category] = correct_counts[category] / total_counts[category]
    
    # 3. 计算统计结果
    matched_categories = list(test_accuracy.items())
    matched_count = len(matched_categories)
    total_count = len(single_categories)
    
    if matched_count > 0:
        avg_accuracy = sum(acc for _, acc in matched_categories) / matched_count
    else:
        avg_accuracy = 0.0
    
    return {
        "average_accuracy": avg_accuracy,
        "total_count": total_count,
        "test_matched_count": matched_count,
        "details": matched_categories
    }

def calculate_high_accuracy_classes_stats(
    test_data: Union[Dict, List[Dict]],
    train_data: Optional[List[Dict]] = None,
    train_index: Optional[Dict] = None,
    accuracy_min: float = 0.8,
    accuracy_max: float = 1.0,
    min_test_count: int = 1
) -> Dict[str, Union[float, int, List[Tuple[str, float, int]]]]:
    """
    统计测试集中高准确率类别在训练集中的出现情况
    
    参数:
        test_data: 测试集数据
        train_data: 训练数据集(与train_index二选一)
        train_index: 预计算训练集索引(与train_data二选一)
        accuracy_max: 准确率最大值
        accuracy_min: 准确率最小值
        min_test_count: 测试集最小出现次数要求
    
    返回:
        {
            "average_train_count": 平均训练集出现次数,
            "total_classes": 符合条件的类别总数,
            "classes_details": [(类别, 测试集准确率, 训练集出现次数)],
            "train_count_distribution": {出现次数: 类别数}  # 训练集出现次数分布
        }
    """
    # 1. 准备训练集统计
    if train_index is None:
        if train_data is None:
            raise ValueError("必须提供train_data或train_index")
        
        train_counter = Counter()
        for item in train_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            for definition in definitions:
                category = f"{name}::{definition}"
                train_counter[category] += 1
    else:
        train_counts = train_index["counts"]
        category = train_index["categories"]
        train_counter = Counter(dict(zip(category, train_counts)))
    
    # 2. 从测试集中提取高准确率类别
    high_accuracy_classes = []
    
    if isinstance(test_data, dict) and "class_wise_accuracy" in test_data:
        # 处理class_wise_accuracy格式
        for category, metrics in test_data["class_wise_accuracy"].items():
            test_count = metrics.get("support", 0)
            accuracy = metrics.get("accuracy", 0.0)
            
            if (accuracy_max >= accuracy >= accuracy_min and 
                test_count >= min_test_count and
                category in train_counter):
                high_accuracy_classes.append((
                    category,
                    accuracy,
                    train_counter[category]
                ))
    
    elif isinstance(test_data, list):
        # 处理原始数据格式(需要计算准确率)
        category_stats = {}
        
        # 首先统计测试集中每个类别的表现
        for item in test_data:
            name = item.get("polysemous", {}).get("name", "")
            definitions = item.get("correct_definitions_in_context", [])
            predicted = item.get("predicted_definition", "")
            
            for definition in definitions:
                category = f"{name}::{definition}"
                if category not in category_stats:
                    category_stats[category] = {
                        "correct": 0,
                        "total": 0
                    }
                
                category_stats[category]["total"] += 1
                if definition == predicted:
                    category_stats[category]["correct"] += 1
        
        # 筛选符合条件的类别
        for category, stats in category_stats.items():
            if stats["total"] >= min_test_count and category in train_counter:
                accuracy = stats["correct"] / stats["total"]
                if accuracy > accuracy_min:
                    high_accuracy_classes.append((
                        category,
                        accuracy,
                        train_counter[category]
                    ))
    
    # 3. 计算统计结果
    total_classes = len(high_accuracy_classes)
    
    if total_classes > 0:
        avg_train_count = sum(count for _, _, count in high_accuracy_classes) / total_classes
        
        # 计算训练集出现次数分布
        count_dist = Counter()
        for _, _, count in high_accuracy_classes:
            count_dist[count] += 1
    else:
        avg_train_count = 0.0
        count_dist = Counter()
    
    return {
        "average_train_count": avg_train_count,
        "total_classes": total_classes,
        "classes_details": sorted(high_accuracy_classes, key=lambda x: -x[1]),  # 按准确率降序
        "train_count_distribution": dict(sorted(count_dist.items()))
    }

def build_frequency_index(data):
    """构建频率索引并返回排序后的类别列表和计数列表"""
    counter = Counter()
    for item in data:
        name = item["polysemous"]["name"]
        for definition in item["correct_definitions_in_context"]:
            category = f"{name}::{definition}"
            counter[category] += 1
    
    # 按频率降序排序
    sorted_categories = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    categories, counts = zip(*sorted_categories) if sorted_categories else ([], [])
    
    return {
        "categories": categories,
        "counts": counts,
        "index": {cat: i for i, cat in enumerate(categories)}
    }

def build_frequency_tree(data):
    """{word:{sense1:proportion, sense2:proportion, ..., count: number}, ...}

    Args:
        data (_type_): _description_
    """
    tree = {}
    for item in data:
        name = item["polysemous"]["name"]
        # definitions = item['polysemous']["sense_definitions_list"]
        current_definitions = item["correct_definitions_in_context"]
        if name not in tree:
            tree[name] = {'count':0}
        tree[name]['count'] += 1
        for definition in current_definitions:
            if definition not in tree[name]:
                tree[name][definition] = 0
            tree[name][definition] += 1
                
    return tree

def save_index(index, file_path):
    """保存频率索引到JSON文件"""
    with open(file_path, 'w') as f:
        json.dump({
            "categories": index["categories"],
            "counts": index["counts"]
        }, f)

def load_index(file_path):
    """从JSON文件加载频率索引"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {
        "categories": data["categories"],
        "counts": data["counts"],
        "index": {cat: i for i, cat in enumerate(data["categories"])}
    }

def word_level_statistics(model_path: str, dataset_name: str, train_data_path: str = "datasets/SemCor/semcor.json"):
    """"offer::present for acceptance or rejection": {
            "accuracy": 0.0,
            "support_in_semcor": 31,
            "candidate_senses_number": 16
        }

    Args:
        result (List[Dict]): _description_
    """
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    tree = build_frequency_tree(train_data)
    with open(f"{model_path}/{dataset_name}_results.json", 'r', encoding='utf-8') as f:
        results = json.load(f)['class_wise_accuracy']
    word_stat = {}
    for key, data in results.items():
        word = key.split("::")[0]
        defn = key.split("::")[1]
        if word not in word_stat:
            word_stat[word] = {
                defn: {
                    "accuracy": data["accuracy"],
                    "support_in_semcor": tree[word].get(defn, 0),
                    f"support_in_{dataset_name}": data['support_in_all'],
                    "candidate_senses_number": data["candidate_senses_number"]
                },
            }
        else:
            word_stat[word][defn] = {
                "accuracy": data["accuracy"],
                "support_in_semcor": tree[word].get(defn, 0),
                f"support_in_{dataset_name}": data['support_in_all'],
                "candidate_senses_number": data["candidate_senses_number"]
            }
    merged_word_stat = {}
    for word, defn_data in word_stat.items():
        merged_word_stat[word] = {
            "senses": defn_data,
            "total_senses_in_semcor": tree[word]['count'] if word in tree else 0,
        }
        if word in tree:
            for defn, count in tree[word].items():
                if defn != 'count' and defn in merged_word_stat[word]['senses']:
                    merged_word_stat[word]['senses'][defn]['proportion_in_semcor'] = count / merged_word_stat[word]['total_senses_in_semcor']
                elif defn != 'count':
                    merged_word_stat[word]['senses'][defn] = {
                        "accuracy": None,
                        "support_in_semcor": count,
                        "candidate_senses_number": len(tree[word]) - 1,
                        "proportion_in_semcor": count / merged_word_stat[word]['total_senses_in_semcor']
                    }
    with open(f"{model_path}/{dataset_name}_word_level_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(merged_word_stat, f, ensure_ascii=False, indent=4)
    pass
if __name__ == "__main__":
    
    main()