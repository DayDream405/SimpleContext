import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import joblib
import random
import gzip
from pathlib import Path

from train_monitor import ParameterMonitor
from multi_layer_model import MultiLayerBERTForWSD, MultiLayerRoBERTaForWSD
from batch_manager import DeviceBatchManager

# 配置参数
class Config:
    MODEL_TYPE = "bert" # bert 或 roberta 或 deberta
    # 模型设置（需提前下载好）
    MODEL_NAME = {
        "bert": "/mnt/zly/SpecializedTtraining/bert-base-uncased",
        "roberta": "roberta-base", 
        "deberta": "deberta-v3-base"  # 添加DeBERTa模型
    }[MODEL_TYPE] # 本地模型路径或名称
    MAX_LENGTH = 128
    BATCH_SIZE = {
        "bert": 256,
        "roberta": 128,
        "deberta": 32  # DeBERTa通常需要更小的batch size
    }[MODEL_TYPE]
 
    LEARNING_RATE = {
        "bert": 1e-4,
        "roberta": 6e-5,
        "deberta": 5e-5  # DeBERTa通常使用较小的学习率
    }[MODEL_TYPE]

    EPOCHS = 20

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = {
        "bert": "saved_models/bert_wsd_model",
        "roberta": "saved_models/roberta_wsd_model",
        "deberta": "saved_models/deberta_wsd_model"  # DeBERTa保存路径
    }[MODEL_TYPE]
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_ID = f'{MODEL_NAME}_resamplewithdistribution5_2class_unified'

    # RoBERTa特定的优化器参数
    WARMUP_RATIO = 0.06
    ADAM_EPSILON = 1e-8
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = {
        "bert": 0.999,
        "roberta": 0.98,
        "deberta": 0.999  # DeBERTa使用标准beta2
    }[MODEL_TYPE]
    WEIGHT_DECAY = {
        "bert": 0.01,
        "roberta": 0.1,
        "deberta": 0.01  # DeBERTa使用标准权重衰减
    }[MODEL_TYPE]

    # 模型架构配置
    NUM_HIDDEN_LAYERS = 12  # 修改这个值来改变层数，默认12层
    HIDDEN_SIZE = 768
    NUM_ATTENTION_HEADS = 12
    INTERMEDIATE_SIZE = 3072 # 3072

    # 多层训练配置
    USE_MULTI_LAYER_TRAINING = False
    LAYER_LEARNING_RATES = [
        6e-5, 6e-5, 6e-5, 6e-5, 6e-5, 6e-5  # 越深的层学习率越高
    ]

# 数据预处理
class DataProcessor:
    @staticmethod
    def load_json_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def build_label_encoder(data, dropped_class_path=None):
        sense_to_key = {}
        for item in data:
            target = item['polysemous']['name']
            for sense in item['polysemous']['sense_definitions_list']:
            # for sense in item['correct_definitions_in_context']:
                sense_to_key[f"{target}::{sense}"] = 1
        if dropped_class_path:
            with open(dropped_class_path, 'r') as f:
                dropped_classes = json.load(f)
            for item in data:
                if all(i not in dropped_classes for i in item['polysemous']['sense_definitions_list']):
                    senses = item['polysemous']['sense_definitions_list']
                    for sense in senses:
                        if f"{item['polysemous']['name']}::{sense}" not in sense_to_key:
                            sense_to_key[f"{item['polysemous']['name']}::{sense}"] = 1
                            if len(sense_to_key) >= Config.CLASS_NUM:
                                break
        print("Unique senses:", len(sense_to_key))
        le = LabelEncoder()
        le.fit(list(sense_to_key.keys()))
        # print(len(sense_to_key))
        return le
    
    @staticmethod
    def process_data(data, label_encoder):
        processed = []
        # 过滤未知标签
        filtered_data = [
            item for item in data
            if all(f"{item['polysemous']['name']}::{sense}" in label_encoder.classes_
            for sense in item['correct_definitions_in_context'])
        ]
        for item in tqdm(filtered_data, desc='processing data', position=0):
            context = item['context']
            target = item['target']
            marked_context = context.replace(target, f"[TGT]{target}[/TGT]")
            
            for correct_def in item['correct_definitions_in_context']:
                label_key = f"{item['polysemous']['name']}::{correct_def}"
                if label_key in label_encoder.classes_:
                    processed.append({
                        'input': marked_context,
                        'label': label_encoder.transform([label_key])[0],
                        'target_word': target
                    })
        return processed

# 数据集类
class WSDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# 训练器
class WSDTrainer:
    def __init__(self, config):
        self.config = config
        if "roberta" in config.MODEL_NAME.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, add_prefix_space=True
        )
        else: # bert 和 deberta
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        # 根据模型类型添加特殊token
        if "roberta" in config.MODEL_NAME.lower():
            # RoBERTa需要添加前缀空格
            self.tokenizer.add_tokens([" [TGT]", " [/TGT]"])
        else:
            # BERT 和 deberta保持原样
            self.tokenizer.add_tokens(["[TGT]", "[/TGT]"])
        
        # 监控配置
        self.monitor_config = {
            'log_dir': 'param_monitor_logs',
            'track_gradients': False,
            'track_updates': False,
            'print_freq': 100,  # 每10步打印一次
            'export_freq': 1000  # 每100步导出一次
        }
        self.monitor = None
        
    
    def _create_layer_optimizers_bert(self):
        """为BERT模型创建分层优化器"""
        optimizers = []
        # 收集从第i层到第0层（第一层）的所有参数
        all_layer_params = []
        for layer_num in range(self.config.NUM_HIDDEN_LAYERS):  # 从第0层到第i层
            if layer_num < len(self.model.bert.encoder.layer):
                layer_module = self.model.bert.encoder.layer[layer_num]
                all_layer_params.append(list(layer_module.parameters()))
        # 为每一层创建独立的优化器(除最后一层)
        for i in range(self.config.NUM_HIDDEN_LAYERS - 1):
            lr = self.config.LAYER_LEARNING_RATES[i] if i < len(self.config.LAYER_LEARNING_RATES) else 6e-5
            layer_params = []
            # 添加BERT编码器从第i层到第0层的参数
            for layer_num in range(i, -1, -1):  # 从第i层到第0层
                if layer_num < len(self.model.bert.encoder.layer):
                    layer_params.extend(all_layer_params[layer_num])
            # 创建优化器
            optimizer = AdamW(
                layer_params,
                lr=lr,
                eps=self.config.ADAM_EPSILON,
            )
            optimizers.append(optimizer)
        # 主分类器优化器
        main_lr = self.config.LAYER_LEARNING_RATES[-1] if self.config.LAYER_LEARNING_RATES else self.config.LEARNING_RATE
        main_optimizer = AdamW(
            self.model.parameters(),
            lr=main_lr,
            eps=self.config.ADAM_EPSILON,
            weight_decay=self.config.WEIGHT_DECAY
        )
        optimizers.append(main_optimizer)
        return optimizers
    
    def _create_layer_optimizers_roberta(self):
        """为RoBERTa模型创建分层优化器"""
        optimizers = []
        
        # 为每一层分类器创建独立的优化器
        for i in range(self.config.NUM_HIDDEN_LAYERS):
            lr = self.config.LAYER_LEARNING_RATES[i] if i < len(self.config.LAYER_LEARNING_RATES) else 1e-5
            optimizer = AdamW(
                self.model.layer_classifiers[i].parameters(),
                lr=lr,
                betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
                eps=self.config.ADAM_EPSILON,
                weight_decay=self.config.WEIGHT_DECAY
            )
            optimizers.append(optimizer)
        
        # 主分类器优化器
        main_lr = self.config.LAYER_LEARNING_RATES[-1] if self.config.LAYER_LEARNING_RATES else self.config.LEARNING_RATE
        main_optimizer = AdamW(
            self.model.classifier.parameters(),
            lr=main_lr,
            betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
            eps=self.config.ADAM_EPSILON,
            weight_decay=self.config.WEIGHT_DECAY
        )
        optimizers.append(main_optimizer)
        
        return optimizers

    def prepare_data(self, train_path, training_artifacts_path=None, original_data_path='datasets/SemCor/semcor.json'):
        train_data = None
        val_data = None
        if training_artifacts_path:
        # 加载保存的训练参数
            print(f"加载训练工件: {training_artifacts_path}")
            train_data, val_data, label_encoder, tokenizer = load_training_artifacts(training_artifacts_path)
            if "roberta" in self.config.MODEL_NAME.lower():
                for item in tqdm(train_data, desc='processing data for roberta', position=0):
                    item['input'] = item['input'].replace("[TGT]", " [TGT]").replace("[/TGT]", " [/TGT]")
                for item in tqdm(val_data, desc='processing data for roberta', position=0):
                    item['input'] = item['input'].replace("[TGT]", " [TGT]").replace("[/TGT]", " [/TGT]")
            else:
                self.tokenizer = tokenizer
        # 加载原始数据
        if not training_artifacts_path:
            print(f"未提供训练工件路径，重新处理数据, train_data_path: {train_path}")
            train_raw = DataProcessor.load_json_data(train_path)
            val_raw = DataProcessor.load_json_data("/mnt/zly/SpecializedTtraining/datasets/en/semeval2007.json")
        original_raw = DataProcessor.load_json_data(original_data_path)
        # 构建统一的标签编码器
        if training_artifacts_path:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = DataProcessor.build_label_encoder(original_raw)
        
        if not training_artifacts_path:
            train_keys = set()  # 使用集合去重
            for item in original_raw:
                for key in item["correct_definition_key"]:
                    train_keys.add(key)  # 添加到集合
            
            # 过滤未知标签
            filtered_val_raw = []
            for item in val_raw:
                # 检查是否有任意一个 correct_definition_key 在训练集中
                if any(key in train_keys for key in item["correct_definition_key"]):
                    filtered_val_raw.append(item)
            
            # 处理数据
            train_data = DataProcessor.process_data(train_raw, self.label_encoder)
            val_data = DataProcessor.process_data(filtered_val_raw, self.label_encoder)
        
            # 保存数据
            save_training_artifacts(train_data, val_data, self.label_encoder, self.tokenizer, f"training_artifacts_{self.config.MODEL_ID}")

        # random.shuffle(train_data)
        # target_size = int(0.75*len(train_data))
        # sub_train_data = train_data[:target_size]
        # filtered_data = train_data[target_size:]
        
        # existing_classes = set()
        # # 记录存在的类别
        # for item in sub_train_data:
        #     label_name = self.label_encoder.inverse_transform([item['label']])[0]
        #     target, sense = label_name.split("::")
        #     existing_classes.add(sense)
        # print(len(existing_classes))
        # # 记录被过滤掉的类别
        # dropped_classes = set()
        # for item in filtered_data:
        #     label_name = self.label_encoder.inverse_transform([item['label']])[0]
        #     target, sense = label_name.split("::")
        #     if sense not in existing_classes:
        #         dropped_classes.add(sense)
        # print(len(dropped_classes))
        # with open(f'{self.config.MODEL_ID}_dropped_classes.json', 'w') as f:
        #     json.dump(list(dropped_classes), f, ensure_ascii=False, indent=4)
        # train_data = sub_train_data
        
        # # 过滤数据
        # with open('bert_same_accuracy0.json', 'r') as f:
        #     retained_classes = json.load(f)
        # train_data = [item for item in train_data if self.label_encoder.inverse_transform([item['label']])[0] not in retained_classes]

        # #关键词替换
        # with open('annotated/keywords_annotated_bert_same_accuracy0_filtered_semcor_5.json', 'r') as f:
        #     annotated_data = json.load(f)
        # count = 0
        # for item in train_data:
        #     target_word = item['target_word']
        #     # 在annotated_data中找到对应key的context
        #     for annotated_item in annotated_data:
        #         annotated_target = annotated_item['target']
        #         original_context = item['input'].replace("[TGT]", "").replace("[/TGT]", "")
        #         if annotated_target == target_word and annotated_item['context'] == original_context:
        #             item['input'] = annotated_item['key_context'].replace(target_word, f"[TGT]{target_word}[/TGT]")
        #             annotated_data.remove(annotated_item)  # 移除已使用的项，避免重复匹配
        #             count += 1
        #             break
        # print(f"关键词替换完成，替换了 {count} 条训练数据。")

        # 创建数据集
        # train_data = random.sample(train_data, min(20000, len(train_data)))  # 随机抽取20k样本
        # #保存抽样后的数据
        # with open(f'{self.config.MODEL_ID}_sampled_train_data.json', 'w', encoding='utf-8') as f:
        #     json.dump(train_data, f, ensure_ascii=False, indent=4)
        train_dataset = WSDDataset(train_data, self.tokenizer, self.config.MAX_LENGTH)
        val_dataset = WSDDataset(val_data, self.tokenizer, self.config.MAX_LENGTH)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        # 初始化模型 - 修改为使用自定义的多层模型
        if "roberta" in self.config.MODEL_NAME.lower():
            from transformers import RobertaConfig, RobertaForSequenceClassification
            model_config = RobertaConfig.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        elif "deberta" in self.config.MODEL_NAME.lower():
            from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
            
            model_config = DebertaV2Config.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import BertConfig, BertForSequenceClassification
            model_config = BertConfig.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.config.DEVICE)
        
        
        self.monitor = ParameterMonitor(self.model, self.monitor_config['log_dir'],
                                          track_gradients=self.monitor_config['track_gradients'],
                                          track_updates=self.monitor_config['track_updates'])
        # 导出初始参数树
        self.monitor.export_parameter_tree(
            os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_ID}_param_tree.json")
        )
    
    def train(self):
        if self.config.USE_MULTI_LAYER_TRAINING:
            # 创建分层优化器
            if "roberta" in self.config.MODEL_NAME.lower():
                optimizers = self._create_layer_optimizers_roberta()
            else:
                optimizers = self._create_layer_optimizers_bert()
            # 学习率调度器
            schedulers = []
            if "roberta" in self.config.MODEL_NAME.lower():
                from transformers import get_linear_schedule_with_warmup
                for i, optimizer in enumerate(optimizers):
                    total_steps = len(self.train_loader) * self.config.EPOCHS
                    warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps
                    )
                    schedulers.append(scheduler)
            else:
                schedulers = [None] * len(optimizers)
        else:
            # 根据模型类型设置优化器参数
            if "roberta" in self.config.MODEL_NAME.lower():
                # 优化器设置
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if not any(nd in n for nd in no_decay)],
                        'weight_decay': self.config.WEIGHT_DECAY,
                    },
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                    },
                ]
                # RoBERTa风格的优化器
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.LEARNING_RATE,
                    betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
                    eps=self.config.ADAM_EPSILON,
                    weight_decay=self.config.WEIGHT_DECAY
                )
            elif "deberta" in self.config.MODEL_NAME.lower():
                # DeBERTa风格的优化器（与BERT类似）
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if not any(nd in n for nd in no_decay)],
                        'weight_decay': self.config.WEIGHT_DECAY,
                    },
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                    },
                ]
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.LEARNING_RATE,
                    eps=self.config.ADAM_EPSILON
                )
            else:
                # BERT风格的优化器
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.LEARNING_RATE,
                    eps=self.config.ADAM_EPSILON
                )
            
            # 学习率调度器（RoBERTa通常需要warmup）
            if "roberta" in self.config.MODEL_NAME.lower() or "deberta" in self.config.MODEL_NAME.lower():
                from transformers import get_linear_schedule_with_warmup
                total_steps = len(self.train_loader) * self.config.EPOCHS
                warmup_steps = int(total_steps * self.config.WARMUP_RATIO)  # 6% warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            else:
                scheduler = None

        best_acc = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_main_loss = 0
            total_layer_losses = [0] * self.config.NUM_HIDDEN_LAYERS
            # # 记录更新前的参数状态
            # global_step = (epoch + 1) * len(self.train_loader)
            # self.monitor.record_pre_update(global_step)

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                # batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
                with DeviceBatchManager(batch, self.config.DEVICE) as batch_on_device:

                # # 记录更新前的参数状态
                # global_step = epoch * len(self.train_loader) + step
                # self.monitor.record_pre_update(global_step)
                    if self.config.USE_MULTI_LAYER_TRAINING:
                        # 前向传播获取各层输出
                        with torch.no_grad():  # 禁用主模型的梯度计算
                            outputs = self.model(
                                # input_ids=batch_on_device['input_ids'],
                                # attention_mask=batch_on_device['attention_mask'],
                                **batch_on_device,
                                output_hidden_states=True
                            )
                            hidden_states = outputs.hidden_states
                        # 分别计算每一层的损失并进行反向传播
                        for layer_idx in range(self.config.NUM_HIDDEN_LAYERS):
                            # 重置梯度
                            optimizers[layer_idx].zero_grad()
                            
                            # 获取该层的分类器输出
                            layer_classifier = self.model.classifier
                            layer_hidden_state = hidden_states[layer_idx + 1][:, 0, :]  # [CLS] token
                            layer_logits = layer_classifier(layer_hidden_state)
                            
                            # 计算该层的损失
                            loss_fct = nn.CrossEntropyLoss()
                            layer_loss = loss_fct(layer_logits.view(-1, self.model.num_labels), 
                                                batch_on_device['labels'].view(-1))
                            
                            # 仅对该层分类器进行反向传播
                            layer_loss.backward()

                            # 更新该层参数
                            optimizers[layer_idx].step()
                            if schedulers[layer_idx]:
                                schedulers[layer_idx].step()
                            
                            total_layer_losses[layer_idx] += layer_loss.item()
                            # 立即清理中间变量
                            del layer_hidden_state, layer_logits, layer_loss, layer_classifier
                        
                        main_loss = total_layer_losses[-1]  # 最后一层的损失
                        total_main_loss += main_loss
                    else:
                        optimizer.zero_grad()
                        outputs = self.model(**batch_on_device)
                        loss = outputs.loss
                        loss.backward()
                        
                        # 梯度裁剪（RoBERTa常用）
                        if "roberta" in self.config.MODEL_NAME.lower() or "deberta" in self.config.MODEL_NAME.lower():
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        total_main_loss += loss.item()
                # 清理基础张量
                if 'hidden_states' in locals():
                    del hidden_states
                del outputs

                # 定期清理GPU缓存
                if step % 30 == 0:
                    torch.cuda.empty_cache()
                # # 记录更新后的参数变化
                # self.monitor.record_post_update(global_step)
                
                        

                # # 定期打印参数变化
                # if global_step % self.monitor_config['print_freq'] == 0:
                #     self.monitor.print_param_changes(global_step)
                
                # # 定期导出监控数据
                # if global_step % self.monitor_config['export_freq'] == 0:
                #     summary = self.monitor.get_param_summary(global_step)
                #     print(f"步骤 {global_step} - 参数摘要:")
                #     print(f"  总参数: {summary['total_parameters']:,}")
                #     print(f"  可训练参数: {summary['trainable_parameters']:,}")
            # # 记录更新后的参数变化
            # self.monitor.record_post_update(global_step)
            # self.monitor.print_param_changes(global_step)
            # 验证
            val_acc = self.evaluate()
            avg_loss = total_main_loss / len(self.train_loader)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            if self.config.USE_MULTI_LAYER_TRAINING:
                # 打印各层损失和梯度信息
                layer_info = []
                for i, layer_loss in enumerate(total_layer_losses):
                    avg_layer_loss = layer_loss / len(self.train_loader)
                    layer_info.append(f"L{i+1}: loss={avg_layer_loss:.4f}")
                print(f"各层训练信息: {', '.join(layer_info)}")
            # # 保存每一轮模型
            # self.save_model(epoch+1, loss=avg_loss)
            # 只保存最后一轮的模型
            if epoch == self.config.EPOCHS - 1:
                self.save_model('final', loss=avg_loss)
        
        # # 训练结束后保存监控数据
        # if self.monitor:
        #     self.monitor.save_monitor_data()
        #     self.monitor.cleanup()
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch)
                _, preds = torch.max(outputs.logits, dim=1)
                correct += (preds == batch['labels']).sum().item()
                total += len(batch['labels'])
        
        return correct / total
    
    def save_model(self, epoch, loss):
        save_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_ID}_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        
        # 确保模型处于eval模式
        self.model.eval()
        
        # 保存完整模型（包含结构和权重）
        self.model.save_pretrained(
            save_path,
            safe_serialization=True  # 生成safetensors文件
        )
        
        # 保存tokenizer（确保包含特殊token）
        self.tokenizer.save_pretrained(save_path)
        
        # 保存标签编码器（添加版本兼容处理）
        joblib.dump(
            self.label_encoder,
            os.path.join(save_path, "label_encoder.joblib"),
            protocol=4  # 兼容Python 3.6+
        )
        
        # 保存训练配置（新增）
        torch.save(
            {
                'epoch': epoch,
                'optimizer_state': self.model.state_dict(),
                'loss': loss,
            },
            os.path.join(save_path, "training_state.pt")
        )
        
        print(f"模型文件列表: {os.listdir(save_path)}")
        print(f"模型已完整保存到 {save_path}")

def robust_json_save(data, path, compress=True):
    """安全保存含数值型数据的JSON（自动处理所有类型）"""
    def _converter(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        elif isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, torch.Tensor):
            return o.cpu().numpy().tolist()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"无法序列化类型: {type(o)}")
        
    json_str = json.dumps(data, default=_converter, indent=2, ensure_ascii=False)
        
    if compress:
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)

def save_training_artifacts(train_data, val_data, label_encoder, tokenizer, save_dir):
    """保存训练所有要素"""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
        
    # 1. 保存主数据（JSON格式）
    robust_json_save(train_data, save_dir/"train_data.json.gz")
    robust_json_save(val_data, save_dir/"val_data.json.gz")
        
    # 2. 保存标签编码器（Joblib格式）
    joblib.dump(label_encoder, save_dir/"label_encoder.joblib")
        
    # 3. 保存Tokenizer
    tokenizer.save_pretrained(save_dir/"tokenizer")
    
    # 4. 保存元数据
    train_metadata = {
        "num_samples": len(train_data),
        "labels": list(label_encoder.classes_),
        "data_schema": {k: type(v).__name__ for k,v in train_data[0].items()}
    }
    with open(save_dir/"metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    val_metadata = {
        "num_samples": len(val_data),
        "labels": list(label_encoder.classes_),
        "data_schema": {k: type(v).__name__ for k,v in val_data[0].items()}
    }
    with open(save_dir/"val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)

def load_training_artifacts(save_dir):
    """加载所有训练要素"""
    save_dir = Path(save_dir)
        
    # 1. 加载主数据
    with gzip.open(save_dir/"train_data.json.gz", 'rt', encoding='utf-8') as f:
        train_data = json.load(f)
    with gzip.open(save_dir/"val_data.json.gz", 'rt', encoding='utf-8') as f:
        val_data = json.load(f)
        
    # 2. 加载其他组件
    label_encoder = joblib.load(save_dir/"label_encoder.joblib")
    tokenizer = AutoTokenizer.from_pretrained(save_dir/"tokenizer")
        
    return train_data, val_data, label_encoder, tokenizer

class WSDPredictor:
    def __init__(self, model_path="./saved_models/best_model"):
        self.path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载标签编码器
        self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))
    
    def predict_with_attention(self, context, target_word, return_attention=True):
        """
        预测并返回注意力分数
        
        Args:
            context: 上下文文本
            target_word: 目标词
            return_attention: 是否返回注意力分数
        
        Returns:
            包含预测结果和注意力分数的字典
        """
        # 标记目标词
        if "roberta" in str(self.model.config.model_type).lower():
            marked_context = context.replace(target_word, f" [TGT]{target_word}[/TGT]")
        else:
            marked_context = context.replace(target_word, f"[TGT]{target_word}[/TGT]")
        
        # 编码输入
        encoding = self.tokenizer(
            marked_context,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                **encoding,
                output_attentions=return_attention,
                output_hidden_states=True
            )
            
            logits = outputs.logits
            pred_id = torch.argmax(logits).item()
            pred_label = self.label_encoder.inverse_transform([pred_id])[0]
        
        # 解析结果
        target, definition = pred_label.split("::", 1)
        confidence = torch.softmax(logits, dim=1).max().item()

        # 获取所有概率
        all_probs = {}
        probabilities = torch.softmax(logits, dim=1)[0]
        for i, class_name in enumerate(self.label_encoder.classes_):
            all_probs[class_name] = probabilities[i].item()
        
        result = {
            'target_word': target,
            'predicted_label': pred_label,
            'predicted_definition': definition,
            'confidence': confidence,
            'context': context,
            'all_probabilities': all_probs
        }
        # print(result['predicted_definition'])
        # 添加注意力分数
        if return_attention and hasattr(outputs, 'attentions'):
            attentions = self._process_attention_scores(
                outputs.attentions, 
                encoding, 
                target_word
            )
            result['attention_scores'] = attentions
        
        # 新增：直接包含聚合注意力摘要
        if 'aggregated_attention' in attentions:
            result['aggregated_attention_summary'] = {
                'top_tokens': attentions['aggregated_attention']['top_attended_tokens'][:10],
                'target_positions': attentions['aggregated_attention']['target_token_info']
            }
        
        return result
    
    def _process_attention_scores(self, attentions, encoding, target_word):
        """
        处理注意力分数，使其更易读
        
        Args:
            attentions: 各层的注意力分数
            encoding: 编码后的输入
            target_word: 目标词
        
        Returns:
            处理后的注意力信息，包含各层和所有层的综合注意力
        """
        if attentions is None:
            return None
        
        # 获取tokenized的文本
        input_ids = encoding['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # 找到目标词的token位置
        target_token_positions = []
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                target_token_positions.append(i)
        
        attention_info = {
            'tokens': tokens,
            'target_token_positions': target_token_positions,
            'layers': [],
            'aggregated_attention': {}  # 新增：聚合注意力信息
        }
        
        # 初始化聚合注意力矩阵（seq_len x seq_len）
        seq_len = len(tokens)
        all_layers_attention_sum = np.zeros((seq_len, seq_len))
        layer_attention_sums = []  # 存储每层的注意力总和
        
        # 处理每一层的注意力
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention形状: (batch_size, num_heads, seq_len, seq_len)
            layer_attention = layer_attention[0]  # 取第一个batch
            num_heads = layer_attention.shape[0]
            
            # 初始化该层的注意力总和矩阵
            layer_attention_sum = np.zeros((seq_len, seq_len))
            
            layer_info = {
                'layer_index': layer_idx,
                'num_heads': num_heads,
                'attention_weights': {},
                'aggregated_target_attention': None  # 新增：该层聚合的目标词注意力
            }
            
            # 初始化该层的目标词注意力向量（从目标词到所有token）
            layer_target_attention = np.zeros(seq_len)
            
            # 对每个注意力头
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx].cpu().numpy()
                
                # 累加到层总和
                layer_attention_sum += head_attention
                
                # 计算目标词token的注意力
                target_attention = []
                for target_pos in target_token_positions:
                    # 目标词对其他token的注意力（行向量）
                    target_to_others = head_attention[target_pos]
                    
                    # 累加到层目标词注意力
                    layer_target_attention += target_to_others
                    
                    target_attention.append({
                        'target_position': target_pos,
                        'target_to_others': target_to_others.tolist(),
                        'others_to_target': head_attention[:, target_pos].tolist()
                    })
                
                layer_info['attention_weights'][f'head_{head_idx}'] = {
                    'target_attention': target_attention,
                    'attention_matrix': head_attention.tolist()
                }
            
            # 存储该层的聚合信息
            layer_info['aggregated_target_attention'] = layer_target_attention.tolist()
            attention_info['layers'].append(layer_info)
            
            # 累加到所有层总和
            all_layers_attention_sum += layer_attention_sum
            layer_attention_sums.append(layer_attention_sum)
        
        # ========== 新增：计算综合注意力信息 ==========
        
        # 1. 计算所有层的目标词注意力总和
        all_layers_target_attention = np.zeros(seq_len)
        for layer_info in attention_info['layers']:
            layer_target_attn = np.array(layer_info['aggregated_target_attention'])
            all_layers_target_attention += layer_target_attn
        
        # 2. 归一化处理（可选，使注意力总和为1）
        if all_layers_target_attention.sum() > 0:
            all_layers_target_attention_norm = all_layers_target_attention / all_layers_target_attention.sum()
        else:
            all_layers_target_attention_norm = all_layers_target_attention
        
        # 3. 找出最关注的token（按注意力值排序）
        token_attention_pairs = []
        for i, token in enumerate(tokens):
            attention_score = all_layers_target_attention[i]
            token_attention_pairs.append({
                'token': token,
                'position': i,
                'attention_score': float(attention_score),
                'normalized_score': float(all_layers_target_attention_norm[i]) if all_layers_target_attention.sum() > 0 else 0.0
            })
        
        # 按注意力值降序排序
        token_attention_pairs.sort(key=lambda x: x['attention_score'], reverse=True)
        
        # 4. 计算每层的归一化目标词注意力（用于比较层间差异）
        layer_normalized_attentions = []
        for layer_idx, layer_info in enumerate(attention_info['layers']):
            layer_target_attn = np.array(layer_info['aggregated_target_attention'])
            if layer_target_attn.sum() > 0:
                norm_attn = layer_target_attn / layer_target_attn.sum()
            else:
                norm_attn = layer_target_attn
            
            layer_normalized_attentions.append({
                'layer_index': layer_idx,
                'normalized_attention': norm_attn.tolist(),
                'top_tokens': []
            })
            
            # 找出该层最关注的token
            layer_top_indices = np.argsort(layer_target_attn)[-10:][::-1]  # 取前10个
            for idx in layer_top_indices:
                if layer_target_attn[idx] > 0:
                    layer_normalized_attentions[-1]['top_tokens'].append({
                        'token': tokens[idx],
                        'position': idx,
                        'attention_score': float(layer_target_attn[idx])
                    })
        
        # 5. 存储聚合的注意力信息
        attention_info['aggregated_attention'] = {
            # 所有层的综合目标词注意力（原始值）
            'all_layers_target_attention': all_layers_target_attention.tolist(),
            
            # 归一化的综合目标词注意力
            'all_layers_target_attention_normalized': all_layers_target_attention_norm.tolist(),
            
            # 最关注的token（按综合注意力排序）
            'top_attended_tokens': token_attention_pairs[:20],  # 取前20个
            
            # 每层的归一化目标词注意力
            'layer_normalized_attentions': layer_normalized_attentions,
            
            # 目标词位置信息
            'target_token_info': [
                {
                    'position': pos,
                    'token': tokens[pos],
                    'is_special_token': tokens[pos] in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
                }
                for pos in target_token_positions
            ]
        }
        
        return attention_info

    def visualize_attention(self, context, target_word, layer_idx=0, head_idx=0):
        """
        可视化特定层和头的注意力
        
        Args:
            context: 上下文文本
            target_word: 目标词
            layer_idx: 层索引
            head_idx: 注意力头索引
        """
        result = self.predict_with_attention(context, target_word)
        
        if 'attention_scores' not in result:
            print("无法获取注意力分数")
            return
        
        attention_info = result['attention_scores']
        tokens = attention_info['tokens']
        
        if layer_idx >= len(attention_info['layers']):
            print(f"层索引 {layer_idx} 超出范围，最大层数: {len(attention_info['layers'])-1}")
            return
        
        layer_info = attention_info['layers'][layer_idx]
        
        if f'head_{head_idx}' not in layer_info['attention_weights']:
            print(f"头索引 {head_idx} 超出范围，最大头数: {layer_info['num_heads']-1}")
            return
        
        head_info = layer_info['attention_weights'][f'head_{head_idx}']
        
        print(f"\n=== 第{layer_idx+1}层 第{head_idx+1}头注意力分析 ===")
        print(f"文本: {context}")
        print(f"目标词: {target_word}")
        print(f"Token化: {' '.join(tokens)}")
        # 保存结果

        # 显示目标词的注意力
        for target_attn in head_info['target_attention']:
            pos = target_attn['target_position']
            print(f"\n目标词位置 {pos} ('{tokens[pos]}') 的注意力:")
            
            # 找到注意力最高的前5个token
            attention_to_others = target_attn['target_to_others']
            top_indices = np.argsort(attention_to_others)[-5:][::-1]
            
            print("  关注最多的token:")
            for idx in top_indices:
                if attention_to_others[idx] > 0.01:  # 只显示显著注意力
                    print(f"    '{tokens[idx]}': {attention_to_others[idx]:.4f}")

    def get_aggregated_attention_analysis(self, context, target_word, top_k=10):
        """
        获取综合注意力分析结果
        
        Args:
            context: 上下文文本
            target_word: 目标词
            top_k: 返回最关注的token数量
        
        Returns:
            包含综合注意力分析的字典
        """
        result = self.predict_with_attention(context, target_word)
        
        if 'attention_scores' not in result or 'aggregated_attention' not in result['attention_scores']:
            return {
                'error': '无法获取聚合注意力信息',
                'prediction': result
            }
        
        attention_info = result['attention_scores']['aggregated_attention']
        
        # 提取关键信息
        analysis = {
            'context': context,
            'target_word': target_word,
            'target_positions': attention_info['target_token_info'],
            'all_probabilities': result['all_probabilities'],  # 添加所有类别的概率信息
            # 所有层的综合注意力
            'aggregated_scores': {
                'raw_scores': attention_info['all_layers_target_attention'],
                'normalized_scores': attention_info['all_layers_target_attention_normalized']
            },
            
            # 最关注的token [:top_k]
            'most_attended_tokens': attention_info['top_attended_tokens'],
            
            # 层间注意力分布
            'layer_attention_distribution': []
        }
        
        # 分析每层的注意力分布
        for layer_attn in attention_info['layer_normalized_attentions']:
            layer_idx = layer_attn['layer_index']
            
            # 计算该层注意力集中在目标词本身的比例
            target_self_attention = 0.0
            for pos_info in attention_info['target_token_info']:
                pos = pos_info['position']
                if pos < len(layer_attn['normalized_attention']):
                    target_self_attention += layer_attn['normalized_attention'][pos]
            
            # 计算该层注意力分布多样性（熵）
            attention_distribution = np.array(layer_attn['normalized_attention'])
            # 避免log(0)错误
            attention_distribution = attention_distribution[attention_distribution > 0]
            if len(attention_distribution) > 0:
                entropy = -np.sum(attention_distribution * np.log(attention_distribution))
                max_entropy = np.log(len(attention_distribution))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0
            
            analysis['layer_attention_distribution'].append({
                'layer': layer_idx,
                'self_attention_rate': float(target_self_attention),
                'attention_entropy': float(normalized_entropy),
                'top_tokens': layer_attn['top_tokens'][:5],  # 每层前5个关注token
            })
        
        return analysis

    def visualize_aggregated_attention(self, context, target_word, top_k=10):
        """
        可视化综合注意力结果
        
        Args:
            context: 上下文文本
            target_word: 目标词
            top_k: 显示的最关注token数量
        """
        analysis = self.get_aggregated_attention_analysis(context, target_word, top_k)
        
        if 'error' in analysis:
            print(f"错误: {analysis['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"综合注意力分析")
        print(f"{'='*60}")
        print(f"上下文: {context}")
        print(f"目标词: {target_word}")
        
        # 显示目标词位置
        target_positions = analysis['target_positions']
        if target_positions:
            positions_str = ', '.join([f"位置 {p['position']} ('{p['token']}')" for p in target_positions])
            print(f"目标词位置: {positions_str}")
        
        print(f"\n{'='*60}")
        print(f"所有层综合注意力 - 最关注的 {top_k} 个token:")
        print(f"{'='*60}")
        
        # 显示最关注的token
        for i, token_info in enumerate(analysis['most_attended_tokens']):
            token = token_info['token']
            pos = token_info['position']
            score = token_info['attention_score']
            norm_score = token_info['normalized_score']
            
            # 标记是否是目标词本身
            is_target = any(pos == p['position'] for p in target_positions)
            target_mark = "[目标词]" if is_target else ""
            
            print(f"{i+1:2d}. 位置 {pos:3d}: '{token:15s}' {target_mark:10s} "
                f"注意力: {score:.6f} (归一化: {norm_score:.4f})")
        
        print(f"\n{'='*60}")
        print(f"层间注意力分布:")
        print(f"{'='*60}")
        
        # 显示每层的注意力特性
        for layer_info in analysis['layer_attention_distribution']:
            layer_idx = layer_info['layer']
            self_rate = layer_info['self_attention_rate']
            entropy = layer_info['attention_entropy']
            
            print(f"\n第 {layer_idx+1:2d} 层:")
            print(f"  - 自我注意力比例: {self_rate:.4f}")
            print(f"  - 注意力分布熵: {entropy:.4f} {'(集中)' if entropy < 0.3 else '(分散)'}")
            
            # 显示该层最关注的token
            if layer_info['top_tokens']:
                top_tokens_str = ', '.join([f"'{t['token']}'" for t in layer_info['top_tokens'][:3]])
                print(f"  - 最关注: {top_tokens_str}")

    def get_layer_attention_summary(self, context, target_word):
        """
        获取各层注意力的摘要信息
        """
        result = self.predict_with_attention(context, target_word)
        
        if 'attention_scores' not in result:
            return None
        
        attention_info = result['attention_scores']
        summary = {
            'context': context,
            'target_word': target_word,
            'num_layers': len(attention_info['layers']),
            'layer_summaries': []
        }
        
        for layer_info in attention_info['layers']:
            layer_idx = layer_info['layer_index']
            num_heads = layer_info['num_heads']
            
            # 计算该层所有头的平均注意力强度
            total_attention_strength = 0
            for head_idx in range(num_heads):
                head_key = f'head_{head_idx}'
                if head_key in layer_info['attention_weights']:
                    for target_attn in layer_info['attention_weights'][head_key]['target_attention']:
                        avg_attention = np.mean(target_attn['target_to_others'])
                        total_attention_strength += avg_attention
            
            avg_layer_attention = total_attention_strength / (num_heads * len(attention_info['target_token_positions']))
            
            summary['layer_summaries'].append({
                'layer_index': layer_idx,
                'average_attention': avg_layer_attention,
                'num_heads': num_heads
            })
        
        return summary

    def predict(self, context, target_word):
        # 标记目标词
        if "roberta" in str(self.model.config.model_type).lower():
            marked_context = context.replace(target_word, f" [TGT]{target_word}[/TGT]")
        else:
            marked_context = context.replace(target_word, f"[TGT]{target_word}[/TGT]")
        
        # 编码输入
        encoding = self.tokenizer(
            marked_context,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(**encoding).logits
            pred_id = torch.argmax(logits).item()
            pred_label = self.label_encoder.inverse_transform([pred_id])[0]
        
        # 解析结果
        target, definition = pred_label.split("::", 1)
        confidence = torch.softmax(logits, dim=1).max().item()

        # 获取所有概率
        all_probs = {}
        probabilities = torch.softmax(logits, dim=1)[0]
        for i, class_name in enumerate(self.label_encoder.classes_):
            all_probs[class_name] = probabilities[i].item()
            
        return {
            'target_word': target,
            'predicted_label': pred_label,
            'confidence': confidence,
            'context': context,
            'all_probabilities': all_probs
        }
    
    def eval_with_clb(self, test_data_path, train_data_path=None):
        """
        每个词义的CLB计算方法：p*l + (1-p)*l_max, p=c/N:采样准确率，对同一个词义N个不同上下文的消歧准确率;l：所有预测正确的上下文平均长度；l_max:所有数据中最长的上下文长度
        """
        # 加载测试数据
        test_data = DataProcessor.load_json_data(test_data_path)
        # 加载模型和相关组件
        tokenizer = self.tokenizer
        label_encoder = self.label_encoder
    
        # 根据模型类型确定特殊token格式
        is_roberta = "roberta" in str(self.model.config.model_type).lower()
    
        # 初始化统计容器
        clb_by_class = defaultdict(list)  # 存储每个词义的CLB值列表
        total_by_class = defaultdict(int)  # 存储每个词义的总样本数量
        context_length_by_class = defaultdict(list)  # 存储每个词义的正确预测上下文长度列表
        max_length = 0  # 存储所有数据中最长的上下文长度
            # 过滤测试数据，确保标签存在;
        filtered_test_data = test_data
        if train_data_path:
            train_data = DataProcessor.load_json_data(train_data_path)

            train_keys = set()  # 使用集合去重
            for item in train_data:
                for key in item["correct_definition_key"]:
                    train_keys.add(key)  # 添加到集合
            # 过滤训练数据
            filtered_test_data = []
            for item in test_data:
                glosses = item['correct_definitions_in_context']
                test_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
                if any(key in train_keys for key in item["correct_definition_key"]):
                    filtered_test_data.append(item)
        
        for item in tqdm(filtered_test_data, desc="评估中"):
            # 根据模型类型处理特殊token
            if is_roberta:
                marked_context = item["context"].replace(item["target"], f" [TGT]{item['target']}[/TGT]")
            else:
                marked_context = item["context"].replace(item["target"], f"[TGT]{item['target']}[/TGT]")
            # 1. 预处理和预测（保持原有逻辑）
            inputs = tokenizer(
                marked_context,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred_id = torch.argmax(logits).item()
            
            predicted_label = label_encoder.inverse_transform([pred_id])[0]
            pred_def = predicted_label.split("::")[1]
            true_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
            true_defs = item["correct_definitions_in_context"]
            
            # 2. 收集统计信息
            for l in true_labels:
                total_by_class[l] += 1
            
            is_correct = pred_def in true_defs
            
            if is_correct:
                context_length_by_class[predicted_label].append(len(item['context']))
            max_length = max(max_length, len(item['context']))
        
        # 3. 计算CLB
        clb_results = {}
        for cls in total_by_class:
            c = len(context_length_by_class[cls])  # 正确预测的数量
            N = total_by_class[cls]  # 总样本数量
            p = c / N if N > 0 else 0  # 采样准确率
            l = np.mean(context_length_by_class[cls]) if context_length_by_class[cls] else 0  # 平均正确预测上下文长度
            l_max = max_length  # 所有数据中最长的上下文长度
            
            clb_value = p * l + (1 - p) * l_max
            clb_results[cls] = {
                'CLB': clb_value,
                'p': p,
                'l': l,
                'l_max': l_max,
                'total_samples': N,
                'correct_samples': c
            }
        return clb_results        
        
    def eval(self, test_data_path, train_data_path=None):
        # 加载测试数据
        test_data = DataProcessor.load_json_data(test_data_path)
        # 加载模型和相关组件
        tokenizer = self.tokenizer
        label_encoder = self.label_encoder

        # 根据模型类型确定特殊token格式
        is_roberta = "roberta" in str(self.model.config.model_type).lower()

        # 初始化统计容器
        correct_by_class = defaultdict(int)
        total_by_class = defaultdict(int)
        # 新增：用于计算F1的容器
        tp_by_class = defaultdict(int)  # 真正例
        fp_by_class = defaultdict(int)  # 假正例
        fn_by_class = defaultdict(int)  # 假反例
        
        accuracy_count = 0
        total_count = 0
        filtered_test_data = test_data
        total_by_class_in_train = defaultdict(int)
        # 过滤测试数据，确保标签存在;
        if train_data_path:
            train_data = DataProcessor.load_json_data(train_data_path)

            train_keys = set()  # 使用集合去重
            for item in train_data:
                for key in item["correct_definition_key"]:
                    train_keys.add(key)  # 添加到集合
            # 过滤训练数据
            # with open('bert-base-uncased_original_0.75semcor_second_unified_dropped_classes.json', 'r') as f:
            #     retained_classes = json.load(f)
            # with open('bert_same_accuracy_other.json', 'r') as f:
            #     retained_classes = json.load(f)
            filtered_test_data = []
            for item in test_data:
                # # 检查是否有任意一个 correct_definition_key 在训练集中
                # if any(key in train_keys for key in item["correct_definition_key"]):
                #     filtered_test_data.append(item)
                glosses = item['correct_definitions_in_context']
                test_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
                # if any(label in retained_classes for label in test_labels) and any(key in train_keys for key in item["correct_definition_key"]):
                # if all(label not in retained_classes for label in glosses) and any(key in train_keys for key in item["correct_definition_key"]):
                if any(key in train_keys for key in item["correct_definition_key"]):
                    filtered_test_data.append(item)
                    
            
            for item in train_data:
                train_labels = [item['polysemous']['name'] + "::" + defn for defn in item['correct_definitions_in_context']]
                for label in train_labels:
                    total_by_class_in_train[label] += 1
                
        candidate_senses_number = defaultdict(int)
        
        for item in tqdm(filtered_test_data, desc="评估中"):
            # 根据模型类型处理特殊token
            if is_roberta:
                marked_context = item["context"].replace(item["target"], f" [TGT]{item['target']}[/TGT]")
            else:
                marked_context = item["context"].replace(item["target"], f"[TGT]{item['target']}[/TGT]")
            # 1. 预处理和预测（保持原有逻辑）
            inputs = tokenizer(
                marked_context,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred_id = torch.argmax(logits).item()
            
            predicted_label = label_encoder.inverse_transform([pred_id])[0]
            pred_def = predicted_label.split("::")[1]
            true_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
            true_defs = item["correct_definitions_in_context"]
            
            # 2. 收集统计信息
            for l in true_labels:
                total_by_class[l] += 1
                candidate_senses_number[l] = len(item['polysemous']['sense_definitions_list'])
            
            # 计算TP, FP, FN
            predicted_class = predicted_label
            is_correct = pred_def in true_defs
            
            # 对于每个真实类别
            for true_label in true_labels:
                if true_label == predicted_class and is_correct:
                    tp_by_class[true_label] += 1
                else:
                    if true_label == predicted_class and not is_correct:
                        fp_by_class[predicted_class] += 1
                    fn_by_class[true_label] += 1
            
            # 对于预测的类别（如果不是真实类别）
            if predicted_class not in true_labels and not is_correct:
                fp_by_class[predicted_class] += 1
            
            if is_correct:
                for true_label in true_labels:
                    correct_by_class[true_label] += 1
                accuracy_count += 1
            total_count += 1
        
        # 3. 计算指标
        def safe_divide(a, b):
            return a / b if b > 0 else 0.0
        
        # 按类别统计准确率和其他指标
        class_metrics = {}
        all_classes = set(list(total_by_class.keys()) + list(tp_by_class.keys()))
        
        # 计算Macro-F1
        f1_scores = []
        
        for label in all_classes:
            label_str = str(label)
            tp = tp_by_class.get(label_str, 0)
            fp = fp_by_class.get(label_str, 0)
            fn = fn_by_class.get(label_str, 0)
            
            # 计算precision, recall, f1
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
            
            acc = safe_divide(correct_by_class.get(label_str, 0), total_by_class.get(label_str, 1))
            class_metrics[label_str] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_in_all": total_by_class.get(label_str, 0),
                "candidate_senses_number": candidate_senses_number.get(label_str, 0),
                "support_in_train": total_by_class_in_train.get(label_str, 0),
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        # 总体准确率
        total_accuracy = accuracy_count / total_count if total_count > 0 else 0.0
        
        # 总体召回率
        total_recall = safe_divide(sum(tp_by_class.values()), sum(tp_by_class.values()) + sum(fn_by_class.values()))
        # Macro-F1：所有类别F1的平均值
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            "class_wise_accuracy": class_metrics,
            "overall_accuracy": total_accuracy,
            "overall_recall": total_recall,
            "macro_f1": macro_f1
        }
# 主函数
def main():
    sample_key = 'bank::financial_institution'
    le = LabelEncoder()
    le.fit([sample_key, 'bank::river_edge'])
    print(le.transform(['bank::river_edge']))

if __name__ == "__main__":
    main()