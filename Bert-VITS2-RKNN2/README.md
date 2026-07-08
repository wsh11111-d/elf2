---
license: agpl-3.0
language:
- zh
pipeline_tag: text-to-speech
---
# Bert-VITS2-RKNN2

RKNN2部署Bert-VITS2文字转语音模型！

- 推理速度：生成512000个样本大概用时2.6秒，速度大概3倍
- 内存占用：约2.3GB

## 使用方法

1. 克隆项目到本地

2. 安装依赖

```bash
# 懒得写requirements.txt了，看rknn_run.py里有什么依赖拿pip安装一下
```

3. 更改你想要生成音频的文字
打开`rknn_run.py`，拉到最下方修改`text`变量
```python
# text = "不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑葚；也不必说鸣蝉在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜向云霄里去了。单是周围的短短的泥墙根一带，就有无限趣味。油蛉在这里低唱， 蟋蟀们在这里弹琴。翻开断砖来，有时会遇见蜈蚣；还有斑蝥，倘若用手指按住它的脊梁，便会“啪”的一声，从后窍喷出一阵烟雾。何首乌藤和木莲藤缠络着，木莲有莲房一般的果实，何首乌有臃肿的根。有人说，何首乌根是有像人形的，吃了便可以成仙，我于是常常拔它起来，牵连不断地拔起来，也曾因此弄坏了泥墙，却从来没有见过有一块根像人样。如果不怕刺，还可以摘到覆盆子，像小珊瑚珠攒成的小球，又酸又甜，色味都比桑葚要好得远。"
text = "我个人认为，这个意大利面就应该拌42号混凝土，因为这个螺丝钉的长度，它很容易会直接影响到挖掘机的扭矩你知道吧。你往里砸的时候，一瞬间它就会产生大量的高能蛋白，俗称ufo，会严重影响经济的发展，甚至对整个太平洋以及充电器都会造成一定的核污染。你知道啊？再者说，根据这个勾股定理，你可以很容易地推断出人工饲养的东条英机，它是可以捕获野生的三角函数的。所以说这个秦始皇的切面是否具有放射性啊，特朗普的N次方是否含有沉淀物，都不影响这个沃尔玛跟维尔康在南极会合。"
```

4. 运行

```bash
python rknn_run.py
```

5. 音频会生成为`output.wav`

## 模型转换

- 转换bert模型: 
  + pytorch转onnx: 执行`optimum-cli export onnx --task feature-extraction --model bert/chinese-roberta-wwm-ext-large/ --output bert/chinese-roberta-wwm-ext-large/model.onnx`
  + onnx转rknn: 参考`bert/chinese-roberta-wwm-ext-large/export_rknn.py`
  + 注意模型的`seq_len`是否与`rknn_run.py`中分词器的`max_length`一致
    ```python
        inputs = tokenizer(text, return_tensors="np",padding="max_length",truncation=True,max_length=256)
    ```
- 转换vits模型: 
  + pytorch转onnx: 参考原项目的`export_onnx.py`
  + onnx转rknn: 参考`onnx/lx/rknn_convert.py`
  + 注意`input_len`是否与`rknn_run.py`中`flow_dec_input_len`的长度一致
  + flow和dec两个模型的执行时间长, 其它模型非常快, 不需要转换
  + flow模型转换后比原onnx模型还慢, 并且貌似模型文件还会明显变大, 不建议转换

## 存在的问题
- 只支持中文
- flow模型没办法有效的使用NPU加速
- 由于NPU只能处理固定长度的输入, 所以需要分割文本, 但是现在貌似还不太清楚怎么做, 有时一句话还没读完就被截断
- 没有实现情感控制等功能
- 其实没必要为了分词器安装一个完整的huggingface Transformers库, 并且还要顺便装一个完全没用的pytorch, 占用2GB空间

## 参考
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [optimum](https://github.com/huggingface/optimum)