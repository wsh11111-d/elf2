import numpy as np
import argparse
import onnxruntime as ort
from rknnlite.api.rknn_lite import RKNNLite
import soundfile as sf
import os
import re
import subprocess
import time
import cn2an
import jieba
import jieba.posseg as psg
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from transformers import AutoTokenizer
from typing import List
from typing import Tuple

def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path


class InferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        ort_config = ort.SessionOptions()
        ort_config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_config.intra_op_num_threads = 4
        ort_config.inter_op_num_threads = 4
        self.enc = ort.InferenceSession(path["enc"], providers=Providers, sess_options=ort_config)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers, sess_options=ort_config)
        self.dp = ort.InferenceSession(path["dp"], providers=Providers, sess_options=ort_config)
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers, sess_options=ort_config)
        # flow模型用onnx比rknn快
        # self.flow = RKNNLite(verbose=False)
        # self.flow.load_rknn(path["flow"])
        # self.flow.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        self.flow = ort.InferenceSession(path["flow"], providers=Providers, sess_options=ort_config)
        self.dec = RKNNLite(verbose=False)
        self.dec.load_rknn(path["dec"])
        self.dec.init_runtime()
        # self.dec = ort.InferenceSession(path["dec"], providers=Providers, sess_options=ort_config)

    def __call__(
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        vqidx,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
        rknn_pad_to = 1024
    ):
        if seq.ndim == 1:
            seq = np.expand_dims(seq, 0)
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)

        start_time = time.time()
        g = self.emb_g.run(
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]
        emb_g_time = time.time() - start_time
        print(f"emb_g 运行时间: {emb_g_time:.4f} 秒")

        g = np.expand_dims(g, -1)
        start_time = time.time()
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),
                "bert_2": bert_en.astype(np.float32),
                "g": g.astype(np.float32),
                # 2.3版本的模型需要注释掉下面两行
                "vqidx": vqidx.astype(np.int64),
                "sid": sid.astype(np.int64),
            },
        )
        enc_time = time.time() - start_time
        print(f"enc 运行时间: {enc_time:.4f} 秒")

        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
        np.random.seed(seed)
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale

        start_time = time.time()
        sdp_output = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0]
        sdp_time = time.time() - start_time
        print(f"sdp 运行时间: {sdp_time:.4f} 秒")

        start_time = time.time()
        dp_output = self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[0]
        dp_time = time.time() - start_time
        print(f"dp 运行时间: {dp_time:.4f} 秒")

        logw = sdp_output * (sdp_ratio) + dp_output * (1 - sdp_ratio)
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )
        #truncate to rknn_pad_to
        actual_len = z_p.shape[2]
        if actual_len > rknn_pad_to:
            print("警告, 输入长度超过 rknn_pad_to, 将被截断")
            z_p = z_p[:,:,:rknn_pad_to]
            y_mask = y_mask[:,:,:rknn_pad_to]
        else:
            z_p = np.pad(z_p, ((0, 0), (0, 0), (0, rknn_pad_to - z_p.shape[2])))
            y_mask = np.pad(y_mask, ((0, 0), (0, 0), (0, rknn_pad_to - y_mask.shape[2])))

        start_time = time.time()
        z = self.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]
        flow_time = time.time() - start_time
        print(f"flow 运行时间: {flow_time:.4f} 秒")

        start_time = time.time()
        dec_output = self.dec.inference([z.astype(np.float32), g])[0]
        dec_time = time.time() - start_time
        print(f"dec 运行时间: {dec_time:.4f} 秒")

        # truncate to actual_len*512
        return dec_output[:,:,:actual_len*512]




class ToneSandhi:
    def __init__(self):
        self.must_neural_tone_words = {
            "麻烦",
            "麻利",
            "鸳鸯",
            "高粱",
            "骨头",
            "骆驼",
            "马虎",
            "首饰",
            "馒头",
            "馄饨",
            "风筝",
            "难为",
            "队伍",
            "阔气",
            "闺女",
            "门道",
            "锄头",
            "铺盖",
            "铃铛",
            "铁匠",
            "钥匙",
            "里脊",
            "里头",
            "部分",
            "那么",
            "道士",
            "造化",
            "迷糊",
            "连累",
            "这么",
            "这个",
            "运气",
            "过去",
            "软和",
            "转悠",
            "踏实",
            "跳蚤",
            "跟头",
            "趔趄",
            "财主",
            "豆腐",
            "讲究",
            "记性",
            "记号",
            "认识",
            "规矩",
            "见识",
            "裁缝",
            "补丁",
            "衣裳",
            "衣服",
            "衙门",
            "街坊",
            "行李",
            "行当",
            "蛤蟆",
            "蘑菇",
            "薄荷",
            "葫芦",
            "葡萄",
            "萝卜",
            "荸荠",
            "苗条",
            "苗头",
            "苍蝇",
            "芝麻",
            "舒服",
            "舒坦",
            "舌头",
            "自在",
            "膏药",
            "脾气",
            "脑袋",
            "脊梁",
            "能耐",
            "胳膊",
            "胭脂",
            "胡萝",
            "胡琴",
            "胡同",
            "聪明",
            "耽误",
            "耽搁",
            "耷拉",
            "耳朵",
            "老爷",
            "老实",
            "老婆",
            "老头",
            "老太",
            "翻腾",
            "罗嗦",
            "罐头",
            "编辑",
            "结实",
            "红火",
            "累赘",
            "糨糊",
            "糊涂",
            "精神",
            "粮食",
            "簸箕",
            "篱笆",
            "算计",
            "算盘",
            "答应",
            "笤帚",
            "笑语",
            "笑话",
            "窟窿",
            "窝囊",
            "窗户",
            "稳当",
            "稀罕",
            "称呼",
            "秧歌",
            "秀气",
            "秀才",
            "福气",
            "祖宗",
            "砚台",
            "码头",
            "石榴",
            "石头",
            "石匠",
            "知识",
            "眼睛",
            "眯缝",
            "眨巴",
            "眉毛",
            "相声",
            "盘算",
            "白净",
            "痢疾",
            "痛快",
            "疟疾",
            "疙瘩",
            "疏忽",
            "畜生",
            "生意",
            "甘蔗",
            "琵琶",
            "琢磨",
            "琉璃",
            "玻璃",
            "玫瑰",
            "玄乎",
            "狐狸",
            "状元",
            "特务",
            "牲口",
            "牙碜",
            "牌楼",
            "爽快",
            "爱人",
            "热闹",
            "烧饼",
            "烟筒",
            "烂糊",
            "点心",
            "炊帚",
            "灯笼",
            "火候",
            "漂亮",
            "滑溜",
            "溜达",
            "温和",
            "清楚",
            "消息",
            "浪头",
            "活泼",
            "比方",
            "正经",
            "欺负",
            "模糊",
            "槟榔",
            "棺材",
            "棒槌",
            "棉花",
            "核桃",
            "栅栏",
            "柴火",
            "架势",
            "枕头",
            "枇杷",
            "机灵",
            "本事",
            "木头",
            "木匠",
            "朋友",
            "月饼",
            "月亮",
            "暖和",
            "明白",
            "时候",
            "新鲜",
            "故事",
            "收拾",
            "收成",
            "提防",
            "挖苦",
            "挑剔",
            "指甲",
            "指头",
            "拾掇",
            "拳头",
            "拨弄",
            "招牌",
            "招呼",
            "抬举",
            "护士",
            "折腾",
            "扫帚",
            "打量",
            "打算",
            "打点",
            "打扮",
            "打听",
            "打发",
            "扎实",
            "扁担",
            "戒指",
            "懒得",
            "意识",
            "意思",
            "情形",
            "悟性",
            "怪物",
            "思量",
            "怎么",
            "念头",
            "念叨",
            "快活",
            "忙活",
            "志气",
            "心思",
            "得罪",
            "张罗",
            "弟兄",
            "开通",
            "应酬",
            "庄稼",
            "干事",
            "帮手",
            "帐篷",
            "希罕",
            "师父",
            "师傅",
            "巴结",
            "巴掌",
            "差事",
            "工夫",
            "岁数",
            "屁股",
            "尾巴",
            "少爷",
            "小气",
            "小伙",
            "将就",
            "对头",
            "对付",
            "寡妇",
            "家伙",
            "客气",
            "实在",
            "官司",
            "学问",
            "学生",
            "字号",
            "嫁妆",
            "媳妇",
            "媒人",
            "婆家",
            "娘家",
            "委屈",
            "姑娘",
            "姐夫",
            "妯娌",
            "妥当",
            "妖精",
            "奴才",
            "女婿",
            "头发",
            "太阳",
            "大爷",
            "大方",
            "大意",
            "大夫",
            "多少",
            "多么",
            "外甥",
            "壮实",
            "地道",
            "地方",
            "在乎",
            "困难",
            "嘴巴",
            "嘱咐",
            "嘟囔",
            "嘀咕",
            "喜欢",
            "喇嘛",
            "喇叭",
            "商量",
            "唾沫",
            "哑巴",
            "哈欠",
            "哆嗦",
            "咳嗽",
            "和尚",
            "告诉",
            "告示",
            "含糊",
            "吓唬",
            "后头",
            "名字",
            "名堂",
            "合同",
            "吆喝",
            "叫唤",
            "口袋",
            "厚道",
            "厉害",
            "千斤",
            "包袱",
            "包涵",
            "匀称",
            "勤快",
            "动静",
            "动弹",
            "功夫",
            "力气",
            "前头",
            "刺猬",
            "刺激",
            "别扭",
            "利落",
            "利索",
            "利害",
            "分析",
            "出息",
            "凑合",
            "凉快",
            "冷战",
            "冤枉",
            "冒失",
            "养活",
            "关系",
            "先生",
            "兄弟",
            "便宜",
            "使唤",
            "佩服",
            "作坊",
            "体面",
            "位置",
            "似的",
            "伙计",
            "休息",
            "什么",
            "人家",
            "亲戚",
            "亲家",
            "交情",
            "云彩",
            "事情",
            "买卖",
            "主意",
            "丫头",
            "丧气",
            "两口",
            "东西",
            "东家",
            "世故",
            "不由",
            "不在",
            "下水",
            "下巴",
            "上头",
            "上司",
            "丈夫",
            "丈人",
            "一辈",
            "那个",
            "菩萨",
            "父亲",
            "母亲",
            "咕噜",
            "邋遢",
            "费用",
            "冤家",
            "甜头",
            "介绍",
            "荒唐",
            "大人",
            "泥鳅",
            "幸福",
            "熟悉",
            "计划",
            "扑腾",
            "蜡烛",
            "姥爷",
            "照顾",
            "喉咙",
            "吉他",
            "弄堂",
            "蚂蚱",
            "凤凰",
            "拖沓",
            "寒碜",
            "糟蹋",
            "倒腾",
            "报复",
            "逻辑",
            "盘缠",
            "喽啰",
            "牢骚",
            "咖喱",
            "扫把",
            "惦记",
        }
        self.must_not_neural_tone_words = {
            "男子",
            "女子",
            "分子",
            "原子",
            "量子",
            "莲子",
            "石子",
            "瓜子",
            "电子",
            "人人",
            "虎虎",
        }
        self.punc = "：，；。？！“”‘’':,;.?!"

    # the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
    # e.g.
    # word: "家里"
    # pos: "s"
    # finals: ['ia1', 'i3']
    def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
        for j, item in enumerate(word):
            if (
                j - 1 >= 0
                and item == word[j - 1]
                and pos[0] in {"n", "v", "a"}
                and word not in self.must_not_neural_tone_words
            ):
                finals[j] = finals[j][:-1] + "5"
        ge_idx = word.find("个")
        if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
            finals[-1] = finals[-1][:-1] + "5"
        elif len(word) >= 1 and word[-1] in "的地得":
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 走了, 看着, 去过
        # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
        #     finals[-1] = finals[-1][:-1] + "5"
        elif (
            len(word) > 1
            and word[-1] in "们子"
            and pos in {"r", "n"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 桌上, 地下, 家里
        elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 上来, 下去
        elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
            finals[-1] = finals[-1][:-1] + "5"
        # 个做量词
        elif (
            ge_idx >= 1
            and (
                word[ge_idx - 1].isnumeric()
                or word[ge_idx - 1] in "几有两半多各整每做是"
            )
        ) or word == "个":
            finals[ge_idx] = finals[ge_idx][:-1] + "5"
        else:
            if (
                word in self.must_neural_tone_words
                or word[-2:] in self.must_neural_tone_words
            ):
                finals[-1] = finals[-1][:-1] + "5"

        word_list = self._split_word(word)
        finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
        for i, word in enumerate(word_list):
            # conventional neural in Chinese
            if (
                word in self.must_neural_tone_words
                or word[-2:] in self.must_neural_tone_words
            ):
                finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
        finals = sum(finals_list, [])
        return finals

    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # e.g. 看不懂
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for i, char in enumerate(word):
                # "不" before tone4 should be bu2, e.g. 不怕
                if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                    finals[i] = finals[i][:-1] + "2"
        return finals

    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # "一" in number sequences, e.g. 一零零, 二一零
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]
        ):
            return finals
        # "一" between reduplication words should be yi5, e.g. 看一看
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        # when "一" is ordinal word, it should be yi1
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # "一" before tone4 should be yi2, e.g. 一段
                    if finals[i + 1][-1] == "4":
                        finals[i] = finals[i][:-1] + "2"
                    # "一" before non-tone4 should be yi4, e.g. 一天
                    else:
                        # "一" 后面如果是标点，还读一声
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        return finals

    def _split_word(self, word: str) -> List[str]:
        word_list = jieba.cut_for_search(word)
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        if first_begin_idx == 0:
            second_subword = word[len(first_subword) :]
            new_word_list = [first_subword, second_subword]
        else:
            second_subword = word[: -len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list

    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        if len(word) == 2 and self._all_tone_three(finals):
            finals[0] = finals[0][:-1] + "2"
        elif len(word) == 3:
            word_list = self._split_word(word)
            if self._all_tone_three(finals):
                #  disyllabic + monosyllabic, e.g. 蒙古/包
                if len(word_list[0]) == 2:
                    finals[0] = finals[0][:-1] + "2"
                    finals[1] = finals[1][:-1] + "2"
                #  monosyllabic + disyllabic, e.g. 纸/老虎
                elif len(word_list[0]) == 1:
                    finals[1] = finals[1][:-1] + "2"
            else:
                finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # e.g. 所有/人
                        if self._all_tone_three(sub) and len(sub) == 2:
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # e.g. 好/喜欢
                        elif (
                            i == 1
                            and not self._all_tone_three(sub)
                            and finals_list[i][0][-1] == "3"
                            and finals_list[0][-1][-1] == "3"
                        ):
                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        finals = sum(finals_list, [])
        # split idiom into two words who's length is 2
        elif len(word) == 4:
            finals_list = [finals[:2], finals[2:]]
            finals = []
            for sub in finals_list:
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                finals += sub

        return finals

    def _all_tone_three(self, finals: List[str]) -> bool:
        return all(x[-1] == "3" for x in finals)

    # merge "不" and the word behind it
    # if don't merge, "不" sometimes appears alone according to jieba, which may occur sandhi error
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        last_word = ""
        for word, pos in seg:
            if last_word == "不":
                word = last_word + word
            if word != "不":
                new_seg.append((word, pos))
            last_word = word[:]
        if last_word == "不":
            new_seg.append((last_word, "d"))
            last_word = ""
        return new_seg

    # function 1: merge "一" and reduplication words in it's left and right, e.g. "听","一","听" ->"听一听"
    # function 2: merge single  "一" and the word behind it
    # if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
    # e.g.
    # input seg: [('听', 'v'), ('一', 'm'), ('听', 'v')]
    # output seg: [['听一听', 'v']]
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        # function 1
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and word == "一"
                and i + 1 < len(seg)
                and seg[i - 1][0] == seg[i + 1][0]
                and seg[i - 1][1] == "v"
            ):
                new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
            else:
                if (
                    i - 2 >= 0
                    and seg[i - 1][0] == "一"
                    and seg[i - 2][0] == word
                    and pos == "v"
                ):
                    continue
                else:
                    new_seg.append([word, pos])
        seg = new_seg
        new_seg = []
        # function 2
        for i, (word, pos) in enumerate(seg):
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] = new_seg[-1][0] + word
            else:
                new_seg.append([word, pos])
        return new_seg

    # the first and the second words are all_tone_three
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and self._all_tone_three(sub_finals_list[i - 1])
                and self._all_tone_three(sub_finals_list[i])
                and not merge_last[i - 1]
            ):
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])

        return new_seg

    def _is_reduplication(self, word: str) -> bool:
        return len(word) == 2 and word[0] == word[1]

    # the last char of first word and the first char of second word is tone_three
    def _merge_continuous_three_tones_2(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and sub_finals_list[i - 1][-1][-1] == "3"
                and sub_finals_list[i][0][-1] == "3"
                and not merge_last[i - 1]
            ):
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])
        return new_seg

    def _merge_er(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and word == "儿" and seg[i - 1][0] != "#":
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg

    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        for i, (word, pos) in enumerate(seg):
            if new_seg and word == new_seg[-1][0]:
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg

    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seg = self._merge_bu(seg)
        try:
            seg = self._merge_yi(seg)
        except:
            print("_merge_yi failed")
        seg = self._merge_reduplication(seg)
        seg = self._merge_continuous_three_tones(seg)
        seg = self._merge_continuous_three_tones_2(seg)
        seg = self._merge_er(seg)
        return seg

    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        finals = self._bu_sandhi(word, finals)
        finals = self._yi_sandhi(word, finals)
        finals = self._neural_sandhi(word, pos, finals)
        finals = self._three_sandhi(word, finals)
        return finals


punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
num_zh_tones = 6

# japanese
ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
num_ja_tones = 2

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
num_en_tones = 4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}




rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text

def get_bert_feature(
    text,
    word2ph,
    style_text=None,
    style_weight=0.7,
):
    global bert_model

    # 使用tokenizer处理输入文本
    inputs = tokenizer(text, return_tensors="np",padding="max_length",truncation=True,max_length=256)
    
    # 运行ONNX模型
    start_time = time.time()
    res = bert_model.inference([inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]])
    flow_time = time.time() - start_time
    print(f"bert 运行时间: {flow_time:.4f} 秒")
    # 处理输出
    # res = np.concatenate(res[0], -1)[0]
    res = res[0][0]
    
    if style_text:
        assert False # TODO
        # style_inputs = tokenizer(style_text, return_tensors="np")
        # style_onnx_inputs = {name: style_inputs[name] for name in bert_model.get_inputs()}
        # style_res = bert_model.run(None, style_onnx_inputs)
        # style_hidden_states = style_res[-1]
        # style_res = np.concatenate(style_hidden_states[-3:-2], -1)[0]
        # style_res_mean = style_res.mean(0)
    
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                # + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T

def clean_text(text, language):
    norm_text = text_normalize(text)
    phones, tones, word2ph = g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    norm_text = text_normalize(text)
    phones, tones, word2ph = g2p(norm_text)
    bert = get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text(text, language_str, style_text=None, style_weight=0.7, add_blank=False):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert_feature(
        norm_text, word2ph, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == "JP":
        bert = np.zeros((1024, len(phone)))
        ja_bert = bert_ori
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == "EN":
        bert = np.zeros((1024, len(phone)))
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = np.array(phone)
    tone = np.array(tone)
    language = np.array(language)
    return bert, ja_bert, en_bert, phone, tone, language


def play_usb_audio(output_path, speaker_device, speaker_volume):
    script_path = Path(__file__).resolve().with_name("play_usb_audio.py")
    command = [
        "python",
        str(script_path),
        str(output_path),
        "--device",
        speaker_device,
        "--volume",
        str(speaker_volume),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError("USB 扬声器播放失败。")


def parse_args():
    parser = argparse.ArgumentParser(description="Bert-VITS2 RKNN 推理并可选播放生成音频。")
    parser.add_argument(
        "--text",
        help="要合成的文本，不传则使用脚本内置示例文本",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="生成音频输出路径，默认 output.wav",
    )
    parser.add_argument(
        "--play-usb",
        action="store_true",
        help="生成完成后自动用 USB 扬声器播放",
    )
    parser.add_argument(
        "--speaker-device",
        default="plughw:4,0",
        help="USB 扬声器 ALSA 设备，默认 plughw:4,0",
    )
    parser.add_argument(
        "--speaker-volume",
        type=float,
        default=1.0,
        help="USB 扬声器软件音量倍率，默认 1.0",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    name = "lx"
    model_prefix = f"onnx/{name}/{name}_"
    bert_path = "./bert/chinese-roberta-wwm-ext-large"
    flow_dec_input_len = 1024
    model_sample_rate = 44100
    # text = "不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑葚；也不必说鸣蝉在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜向云霄里去了。单是周围的短短的泥墙根一带，就有无限趣味。油蛉在这里低唱， 蟋蟀们在这里弹琴。翻开断砖来，有时会遇见蜈蚣；还有斑蝥，倘若用手指按住它的脊梁，便会“啪”的一声，从后窍喷出一阵烟雾。何首乌藤和木莲藤缠络着，木莲有莲房一般的果实，何首乌有臃肿的根。有人说，何首乌根是有像人形的，吃了便可以成仙，我于是常常拔它起来，牵连不断地拔起来，也曾因此弄坏了泥墙，却从来没有见过有一块根像人样。如果不怕刺，还可以摘到覆盆子，像小珊瑚珠攒成的小球，又酸又甜，色味都比桑葚要好得远。"
    text = args.text or "我个人认为，这个意大利面就应该拌42号混凝土，因为这个螺丝钉的长度，它很容易会直接影响到挖掘机的扭矩你知道吧。你往里砸的时候，一瞬间它就会产生大量的高能蛋白，俗称ufo，会严重影响经济的发展，甚至对整个太平洋以及充电器都会造成一定的核污染。你知道啊？再者说，根据这个勾股定理，你可以很容易地推断出人工饲养的东条英机，它是可以捕获野生的三角函数的。所以说这个秦始皇的切面是否具有放射性啊，特朗普的N次方是否含有沉淀物，都不影响这个沃尔玛跟维尔康在南极会合。"

    global bert_model,tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = RKNNLite(verbose=False)
    bert_model.load_rknn(bert_path + "/model.rknn")
    bert_model.init_runtime()
    model = InferenceSession({
        "enc": model_prefix + "enc_p.onnx",
        "emb_g": model_prefix + "emb.onnx",
        "dp": model_prefix + "dp.onnx",
        "sdp": model_prefix + "sdp.onnx",
        "flow": model_prefix + "flow.onnx",
        "dec": model_prefix + "dec.rknn",
    })

    # 从句号分割
    text_seg = re.split(r'(?<=[。！？；])', text)
    output_acc = np.array([0.0])

    for text in text_seg:
        bert, ja_bert, en_bert, phone, tone, language = get_text(text, "ZH", add_blank=True)
        bert = np.transpose(bert)
        ja_bert = np.transpose(ja_bert)
        en_bert = np.transpose(en_bert)

        sid = np.array([0])
        vqidx = np.array([0])

        output = model(phone, tone, language, bert, ja_bert, en_bert, vqidx, sid ,
                       rknn_pad_to=flow_dec_input_len,
                       seed=114514,
                       seq_noise_scale=0.8,
                       sdp_noise_scale=0.6,
                       length_scale=1,
                       sdp_ratio=0,
                       )[0,0]
        output_acc = np.concatenate([output_acc, output])
        print(f"已生成长度: {len(output_acc) / model_sample_rate:.2f} 秒")
        
    output_path = Path(args.output).expanduser().resolve()
    sf.write(str(output_path), output_acc, model_sample_rate)
    print(f"已生成 {output_path}")

    if args.play_usb:
        play_usb_audio(output_path, args.speaker_device, args.speaker_volume)
