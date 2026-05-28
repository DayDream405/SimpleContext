from training import *
from predict import *

def predictor_layer_analysis(predictor, test_data_path, train_data_path=None, layer_num=12):
    """记录模型最终预测标签和真实标签在每一层的概率"""
    EACH_LAYER_CONFIGS = [ {'layer_pattern': f'encoder.layer.{i}', 'strategy': 'identity', 'strength': 1.0} for i in range(layer_num)]

    # 加载测试数据
    test_data = DataProcessor.load_json_data(test_data_path)
    filtered_test_data = test_data 
    # 过滤测试数据，确保标签存在
    if train_data_path:
        train_data = DataProcessor.load_json_data(train_data_path)
        train_keys = set()  # 使用集合去重
        for item in train_data:
            for key in item["correct_definition_key"]:
                train_keys.add(key)  # 添加到集合
        filtered_test_data = []
        for item in test_data:
            # 检查是否有任意一个 correct_definition_key 在训练集中
            if any(key in train_keys for key in item["correct_definition_key"]):
                filtered_test_data.append(item)
    baseline_positive_results = []
    baseline_negative_results = []
    for item in tqdm(filtered_test_data, desc="BaseLine"):
        result = predictor.predict(item['context'], item['target'])
        true_labels = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        predicted_label = result['predicted_label']
        if predicted_label in true_labels:
            baseline_positive_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'max_probabilities': max(result['all_probabilities'].values())
            })
        else:
            baseline_negative_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'true_labels': true_labels,
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'true_probabilities': max([result['all_probabilities'][true_label] for true_label in true_labels]),
                'max_probabilities': max(result['all_probabilities'].values())
            })
        del(result)
    del filtered_test_data

    positive_results_per_layer = []
    negative_results_per_layer = []
    for i in range(layer_num - 1, 0, -1):
        layer_config = EACH_LAYER_CONFIGS[i:layer_num]
        predictor.set_layer_mask(layer_config)
        positive_layer_results = []
        for item in tqdm(baseline_positive_results, desc="Positive Item"):
            result = predictor.predict(item['context'], item['target'])
            positive_layer_results.append({
                'target_word': item['target_word'],
                'predicted_label': item['predicted_label'],
                'predicted_probabilities': result['all_probabilities'][item['predicted_label']],
                'max_probabilities': max(result['all_probabilities'].values())
            })
            del result
        positive_results_per_layer.append(positive_layer_results)
        negative_layer_results = []
        for item in tqdm(baseline_negative_results, desc="Negative Item"):
            result = predictor.predict(item['context'], item['target'])
            negative_layer_results.append({
                'target_word': item['target_word'],
                'predicted_label': item['predicted_label'],
                'predicted_probabilities': result['all_probabilities'][item['predicted_label']],
                'true_probabilities': max([result['all_probabilities'][true_label] for true_label in item['true_labels']]),
                'max_probabilities': max(result['all_probabilities'].values())
            })
            del result
        negative_results_per_layer.append(negative_layer_results)
    positive_prob_per_layer = []
    negative_prob_per_layer = []
    for layer_results in positive_results_per_layer:
        layer_predicted_label_probs = [item['predicted_probabilities'] for item in layer_results]
        mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
        layer_max_probs = [item['max_probabilities'] for item in layer_results]
        mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
        positive_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 
                                        'mean_max_probabilities': mean_max_prob})
    positive_prob_per_layer.reverse()
    layer_predicted_label_probs = [item['predicted_probabilities'] for item in baseline_positive_results]
    mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
    layer_max_probs = [item['max_probabilities'] for item in baseline_positive_results]
    mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
    positive_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob,
                                    'mean_max_probabilities': mean_max_prob})
    for layer_results in negative_results_per_layer:
        layer_predicted_label_probs = [item['predicted_probabilities'] for item in layer_results]
        mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
        layer_true_label_probs = [item['true_probabilities'] for item in layer_results]
        mean_true_prob = sum(layer_true_label_probs) / len(layer_true_label_probs)
        layer_max_probs = [item['max_probabilities'] for item in layer_results]
        mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
        negative_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 'mean_true_prob': mean_true_prob,
                                        'mean_max_probabilities': mean_max_prob})
    negative_prob_per_layer.reverse()
    layer_predicted_label_probs = [item['predicted_probabilities'] for item in baseline_negative_results]
    mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
    layer_true_label_probs = [item['true_probabilities'] for item in baseline_negative_results]
    mean_true_prob = sum(layer_true_label_probs) / len(layer_true_label_probs)
    layer_max_probs = [item['max_probabilities'] for item in baseline_negative_results]
    mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
    negative_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 'mean_true_prob': mean_true_prob,
                                    'mean_max_probabilities': mean_max_prob})
    return {
        'positive_probs': positive_prob_per_layer,
        'negative_probs': negative_prob_per_layer
    }

def predictor_layer_attention_analysis(predictor, baseline_result_path):
    """记录模型最终预测标签和真实标签在每一层的注意力分布"""

    # 加载baseline结果
    DataProcessor.load_json_data(baseline_result_path)
    with open(baseline_result_path, 'r', encoding='utf-8') as f:
        baseline_results = json.load(f)
    baseline_positive_results = baseline_results['positive_items']
    baseline_negative_results = baseline_results['negative_items']
    # # 从回答正确的样本中抽取1个数据
    # sample_positive_item = random.sample(baseline_positive_results, min(1, len(baseline_positive_results)))
    # predictor.visualize_aggregated_attention(sample_positive_item[0]['context'], sample_positive_item[0]['target'])
    # 从回答错误的样本中抽取1个数据
    sample_negative_item = random.sample(baseline_negative_results, min(1, len(baseline_negative_results)))
    # predictor.visualize_aggregated_attention(sample_negative_item[0]['context'], sample_negative_item[0]['target'])
    # Some of the homeless , obviously , had pre-existing mental illness or addiction
    # go through (mental or physical states or experiences); have or possess, either in a concrete or an abstract sense; suffer from; be ill with
    predictor.visualize_aggregated_attention("The stranger really had nothing to do with it , of course", 'do')

def baseline_evaluation(predictor, test_data_path, train_data_path=None):
    """记录模型baseline预测结果"""
    # 加载测试数据
    test_data = DataProcessor.load_json_data(test_data_path)
    filtered_test_data = test_data 
    # 过滤测试数据，确保标签存在
    if train_data_path:
        train_data = DataProcessor.load_json_data(train_data_path)
        train_keys = set()  # 使用集合去重
        for item in train_data:
            for key in item["correct_definition_key"]:
                train_keys.add(key)  # 添加到集合
        filtered_test_data = []
        for item in test_data:
            # 检查是否有任意一个 correct_definition_key 在训练集中
            if any(key in train_keys for key in item["correct_definition_key"]):
                filtered_test_data.append(item)
    baseline_positive_results = []
    baseline_negative_results = []
    for item in tqdm(filtered_test_data, desc="BaseLine"):
        result = predictor.predict(item['context'], item['target'])
        true_labels = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        predicted_label = result['predicted_label']
        if predicted_label in true_labels:
            baseline_positive_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'max_probabilities': max(result['all_probabilities'].values())
            })
        else:
            baseline_negative_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'true_labels': true_labels,
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'true_probabilities': max([result['all_probabilities'][true_label] for true_label in true_labels]),
                'max_probabilities': max(result['all_probabilities'].values())
            })
        del(result)
    del filtered_test_data
    # 保存baseline结果
    baseline_result = {'positive_items': baseline_positive_results,
                       'negative_items': baseline_negative_results}
    path = predictor.path + '/baseline_results.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(baseline_result, f, ensure_ascii=False, indent=4)
    pass

def attention_rank(predictor, k=10):
    with open('/mnt/zly/SpecializedTtraining/annotated/annotated_data.json', 'r', encoding='utf-8') as f:
        annotated_data = json.load(f)
    save_data = []
    acc1_data = []
    acc0_data = []
    for item in tqdm(annotated_data, desc="Attention Rank Evaluation"):
        acc = item['accuracy']
        
        analysis = predictor.get_aggregated_attention_analysis(item['context'], item['target'])
        if acc:
            acc1_data.append({
                'context': item['context'],
                'target': item['target'],
                'most_attended_tokens': analysis['most_attended_tokens'][:k],
                'keywords': item['keywords']
            })
        else:
            acc0_data.append({
                'context': item['context'],
                'target': item['target'],
                'most_attended_tokens': analysis['most_attended_tokens'][:k],
                'keywords': item['keywords']
            })
    save_data.append({
        'accuracy1': acc1_data,
        'accuracy0': acc0_data
        })
       
    with open(predictor.path + '/attention_rank.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    pass

def hr(predector, k = 10):
    with open('annotated/annotated_data.json', 'r', encoding='utf-8') as f:
        annotated_data = json.load(f)
    special_tokens = {'.':[], ',':[], '[pad]':[], '[cls]':[], '[sep]':[], '[unk]':[], '[tgt]':[], '[/tgt]':[]}
    ks0={'keyword1':[], 'keyword2':[], 'keyword3':[]}
    ks1={'keyword1':[], 'keyword2':[], 'keyword3':[]}
    wks0={'keyword1':0, 'keyword2':0, 'keyword3':0}
    wks1={'keyword1':0, 'keyword2':0, 'keyword3':0}
    N0 = 0
    N1 = 0
    from tqdm import tqdm
    for item in tqdm(annotated_data, desc="HR Evaluation"):
        acc = item['accuracy']
        if acc:
            N0 += 1
        else:
            N1 += 1
        keywords = item['keywords']
        analysis = predector.get_aggregated_attention_analysis(item['context'], item['target'])
        for i, token_info in enumerate(analysis['most_attended_tokens']):
            token = token_info['token']
            if token.lower() in special_tokens:
                special_tokens[token.lower()].append(i + 1)
                continue
            for j, keyword in enumerate(keywords):
                if token.lower() in keyword.lower():
                    if acc:
                        ks0[f'keyword{j+1}'].append(i + 1)
                        wks0[f'keyword{j+1}'] += 1
                    else:
                        ks1[f'keyword{j+1}'].append(i + 1)
                        wks1[f'keyword{j+1}'] += 1
    N = len(annotated_data)
    acc0_hr1 = sum([max(0, k - k1 + 1) for k1 in ks0['keyword1']]) / N0 / k
    acc0_hr2 = sum([max(0, k - k2 + 1) for k2 in ks0['keyword2']]) / N0 / k
    acc0_hr3 = sum([max(0, k - k3 + 1) for k3 in ks0['keyword3']]) / N0 / k
    acc1_hr1 = sum([max(0, k - k1 + 1) for k1 in ks1['keyword1']]) / N1 / k
    acc1_hr2 = sum([max(0, k - k2 + 1) for k2 in ks1['keyword2']]) / N1 / k
    acc1_hr3 = sum([max(0, k - k3 + 1) for k3 in ks1['keyword3']]) / N1 / k
    acc0_whr0 = wks0['keyword1'] / N0
    acc0_whr1 = wks0['keyword2'] / N0
    acc0_whr2 = wks0['keyword3'] / N0
    acc1_whr0 = wks1['keyword1'] / N1
    acc1_whr1 = wks1['keyword2'] / N1
    acc1_whr2 = wks1['keyword3'] / N1
    print(f"Weighted Accuracy 0: WHR@{k} for keyword1: {acc0_whr0}, keyword2: {acc0_whr1}, keyword3: {acc0_whr2}")
    print(f"Weighted Accuracy 1: WHR@{k} for keyword1: {acc1_whr0}, keyword2: {acc1_whr1}, keyword3: {acc1_whr2}")
    print(f"Accuracy 0: HR@{k} for keyword1: {acc0_hr1}, keyword2: {acc0_hr2}, keyword3: {acc0_hr3}")
    print(f"Accuracy 1: HR@{k} for keyword1: {acc1_hr1}, keyword2: {acc1_hr2}, keyword3: {acc1_hr3}")
    special_tokens_average_positions = {}
    for token, positions in special_tokens.items():
        if positions:
            average_position = sum(positions) / len(positions)
        else:
            average_position = -1
        special_tokens_average_positions[token] = average_position
    print(f"Special Tokens Attention Average Positions:", special_tokens_average_positions)
    pass

def simple_context(predictor, k=10):
    with open('/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('bert_same_accuracy0.json', 'r', encoding='utf-8') as f:
        same_acc0_data = json.load(f)
    special_tokens = ['.', ',', '[pad]', '[cls]', '[sep]', '[unk]', '[tgt]', '[/tgt]']
    simple_data = []
    remaining_data = []
    from tqdm import tqdm
    for item in tqdm(data, desc="Simple Context"):
        keys = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        
        if any(key in same_acc0_data for key in keys):
            # 获取注意力分析
            analysis = predictor.get_aggregated_attention_analysis(item['context'], item['target'])
            
            # 收集注意力分数最高的词（排除特殊token）
            attended_tokens = set()
            target_token = item['target'].lower()
            
            for token_info in analysis['most_attended_tokens']:
                token = token_info['token']
                # 清理token（移除分词器的特殊标记如##）
                clean_token = token.replace('##', '').lower()
                if clean_token not in special_tokens and clean_token.strip():
                    attended_tokens.add(clean_token)
            
            # 确保目标词始终包含在内
            attended_tokens.add(target_token)
            
            # 从原上下文中提取词，只保留注意力分数高的词
            # 使用正则表达式分割，保留标点符号作为独立token
            import re
            original_tokens = re.findall(r'\w+|[.,!?;:]', item['context'])
            
            # 只保留注意力分数高的词
            preserved_tokens = []
            for token in original_tokens:
                clean_token = token.lower()
                if clean_token in attended_tokens:
                    preserved_tokens.append(token)
            
            # 重建上下文
            scontext = ' '.join(preserved_tokens)
            
            # print("Original Context:", item['context'])
            # print("Preserved Tokens:", preserved_tokens)
            # print("New Context:", scontext)
            
            item['context'] = scontext
    
        simple_data.append(item)
        # else:
        #     remaining_data.append(item)
    with open('subeval2007_acc0.json', 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, ensure_ascii=False, indent=4)
    # with open('remaining_subeval2007_acc0.json', 'w', encoding='utf-8') as f:
    #     json.dump(remaining_data, f, ensure_ascii=False, indent=4)       

def subset_accuracy(subset_path, result_path):
    with open(subset_path, 'r', encoding='utf-8') as f:
        subset_data = json.load(f)
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)['sense_accuracy']
    with open('/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    key_map = {}
    for item in full_data:
        keys = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        for key in keys:
            if key not in key_map:
                key_map[key] = item['correct_definition_key'][0]
    
    total_acc = 0
    count = 0
    print(type(result_data))
    for k1 in subset_data:
        k2 =  key_map.get(k1, None)
        if k2 is None:
            continue
        acc = result_data[k2]
        total_acc += acc
        count += 1
    accuracy = total_acc / count if count > 0 else 0
    print(f"Subset Accuracy: {accuracy:.4f} ({total_acc}/{count})")

import spacy
def extract_dependency_context(text, target_word):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    for token in doc:
        if token.text == target_word:
            # 获取依存子树
            subtree = [t.text for t in token.subtree]
            return ' '.join(subtree)
    return text
def extract_dependency_context_v2(text, target_word, max_depth=1):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    def get_dependents(token, depth=0, max_depth=max_depth):
        """递归获取指定深度的依存词"""
        if depth > max_depth:
            return []
        
        result = [token.text]
        for child in token.children:
            result.extend(get_dependents(child, depth+1, max_depth))
        return result
    
    for token in doc:
        if token.text == target_word:
            # 只获取目标词及其直接/间接依存
            context_words = get_dependents(token)
            return ' '.join(context_words)
    return text

def simple_dataset_context(data_path, name, filter_keys_path=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filter_keys = []
    if filter_keys_path:
        with open(filter_keys_path, 'r', encoding='utf-8') as f:
            filter_dict = json.load(f)
        
        for k, v in filter_dict.items():
            if v == 0:
                filter_keys.append(k)
    for item in data:
        # keys = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        keys = item['correct_definition_key']
        if filter_keys_path and not any(key in filter_keys for key in keys):
            continue

        context = item['context']
        target = item['target']
        # new_context = extract_dependency_context(context, target)
        new_context = extract_dependency_context(context,target)
        item['context'] = new_context
    with open(f'simple_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        

def main():
    # predictor = MaskableWSDPredictor('/mnt/zly/SpecializedTtraining/saved_models/best_model_original_unified_10')
    # simple_context(predictor)
    # subset_accuracy('bert_same_accuracy1.json', 'semeval2007_sense_accuracy.json')
    # simple_dataset_context('/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', 'semeval2007_nofilter')
    # result = predictor.eval('subeval2007_03.json')
    # with open('subeval2007_03_result.json', 'w', encoding='utf-8') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)

    # attention_rank(predictor)

    # config = Config()
    # trainer = WSDTrainer(config)
    # trainer.prepare_data('semcor_new_subset_global_dist_5_2class.json'
    #                     #  ,'training_artifacts_/mnt/zly/SpecializedTtraining/bert-base-uncased_samplewithdistribution_unified'
    #                     , original_data_path='semcor_new_subset_global_dist_5_2class.json'
    #                      )
    # trainer.train()

    predictor = WSDPredictor('/mnt/zly/SpecializedTtraining/saved_models/best_model_original_unified_10')
    # clbs = predictor.eval_with_clb('/mnt/zly/SpecializedTtraining/datasets/en/all.json', '/mnt/zly/SpecializedTtraining/datasets/SemCor/semcor.json')
    # with open(predictor.path + '/clb_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(clbs, f, ensure_ascii=False, indent=4)
    metrics = predictor.eval('simple_semeval2007_nofilter.json', '/mnt/zly/SpecializedTtraining/datasets/SemCor/semcor.json')
    print(f"acc:{metrics['overall_accuracy']}, recall:{metrics['overall_recall']},f1:{metrics['macro_f1']}")
    path = predictor.path + '/simple07_results.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # predictor = MaskableWSDPredictor('/mnt/zly/SpecializedTtraining/bert-base-uncased_resamplewithdistribution5_2class_unified_final')
    # results = predictor_layer_analysis(predictor, '/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json', 'semcor_new_subset_global_dist_5_2class.json', layer_num=12)
    # path = predictor.path + '/mean_prob_per_layer.json'
    # results_dict = {'positive items': {f'layer.{i}': result for i, result in enumerate(results['positive_probs'])},
    #                 'negative items': {f'layer.{i}': result for i, result in enumerate(results['negative_probs'])}}
    # with open(path, 'w', encoding='utf-8') as f:
    #     json.dump(results_dict, f, ensure_ascii=False, indent=4)
    # context = extract_dependency_context('Homeless people not only lack safety, privacy and shelter, they also lack the elementary necessities of nutrition, cleanliness and basic health care.', 'people')
    # print(context)
if __name__ == "__main__":
    main()