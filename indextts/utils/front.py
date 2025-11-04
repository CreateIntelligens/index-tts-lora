# -*- coding: utf-8 -*-
import os
import traceback
import re
from typing import List, Union, overload
import warnings
from indextts.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor


class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
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
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }

    def match_email(self, email):
        # 正則表示式匹配郵箱格式：數字英文@數字英文.英文
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    匹配拼音聲調格式：pinyin+數字，聲調1-5，5表示輕聲
    例如：xuan4, jve2, ying1, zhong4, shang5
    不匹配：beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    匹配人名，格式：中文·中文，中文·中文-中文
    例如：克里斯托弗·諾蘭，約瑟夫·高登-萊維特
    """

    # 匹配常見英語縮寫 's，僅用於替換為 is，不匹配所有 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"


    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() == "Darwin":
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())
            
            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 恢復人名
            result = self.restore_names(result, original_name_list)
            # 恢復拼音聲調
            result = self.restore_pinyin_tones(result, pinyin_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                result = self.en_normalizer.normalize(text)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        將 jqx 的韻母為 u/ü 的拼音轉換為 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 匹配 jqx 的韻母為 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        替換人名為佔位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·諾蘭 -> <n_a>
        """
        # 人名
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 替換佔位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        恢復人名為原來的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替換為佔位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        """
        替換拼音聲調為佔位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 聲母韻母+聲調數字
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替換為佔位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢復拼音中的音調數字（1-5）為原來的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替換佔位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 載入詞表
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 預處理器
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # 預處理
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        # 預處理
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_sentences_by_token(
        tokenized_str: List[str], split_tokens: List[str], max_tokens_per_sentence: int
    ) -> List[List[str]]:
        """
        將tokenize後的結果按特定token進一步分割
        """
        # 處理特殊情況
        if len(tokenized_str) == 0:
            return []
        sentences: List[List[str]] = []
        current_sentence = []
        current_sentence_tokens_len = 0
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_sentence.append(token)
            current_sentence_tokens_len += 1
            if current_sentence_tokens_len <= max_tokens_per_sentence:
                if token in split_tokens and current_sentence_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            # 後續token是'，則不切分
                            current_sentence.append(tokenized_str[i + 1])
                            i += 1
                    sentences.append(current_sentence)
                    current_sentence = []
                    current_sentence_tokens_len = 0
                continue
            # 如果當前tokens的長度超過最大限制
            if not  ("," in split_tokens or "▁," in split_tokens ) and ("," in current_sentence or "▁," in current_sentence): 
                # 如果當前tokens中有,，則按,分割
                sub_sentences = TextTokenizer.split_sentences_by_token(
                    current_sentence, [",", "▁,"], max_tokens_per_sentence=max_tokens_per_sentence
                )
            elif "-" not in split_tokens and "-" in current_sentence:
                # 沒有,，則按-分割
                sub_sentences = TextTokenizer.split_sentences_by_token(
                    current_sentence, ["-"], max_tokens_per_sentence=max_tokens_per_sentence
                )
            else:
                # 按照長度分割
                sub_sentences = []
                for j in range(0, len(current_sentence), max_tokens_per_sentence):
                    if j + max_tokens_per_sentence < len(current_sentence):
                        sub_sentences.append(current_sentence[j : j + max_tokens_per_sentence])
                    else:
                        sub_sentences.append(current_sentence[j:])
                warnings.warn(
                    f"The tokens length of sentence exceeds limit: {max_tokens_per_sentence}, "
                    f"Tokens in sentence: {current_sentence}."
                    "Maybe unexpected behavior",
                    RuntimeWarning,
                )
            sentences.extend(sub_sentences)
            current_sentence = []
            current_sentence_tokens_len = 0
        if current_sentence_tokens_len > 0:
            assert current_sentence_tokens_len <= max_tokens_per_sentence
            sentences.append(current_sentence)
        # 如果相鄰的句子加起來長度小於最大限制，則合併
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            if len(merged_sentences) == 0:
                merged_sentences.append(sentence)
            elif len(merged_sentences[-1]) + len(sentence) <= max_tokens_per_sentence:
                merged_sentences[-1] = merged_sentences[-1] + sentence
            else:
                merged_sentences.append(sentence)
        return merged_sentences

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "▁.",
        # "▁!", # unk
        "▁?",
        "▁...", # ellipsis
    ]
    def split_sentences(self, tokenized: List[str], max_tokens_per_sentence=120) -> List[List[str]]:
        return TextTokenizer.split_sentences_by_token(
            tokenized, self.punctuation_marks_tokens, max_tokens_per_sentence=max_tokens_per_sentence
        )


if __name__ == "__main__":
    # 測試程式

    text_normalizer = TextNormalizer()

    cases = [
        "IndexTTS 正式釋出1.0版本了，效果666",
        "暈XUAN4是一種GAN3覺",
        "我愛你！",
        "I love you!",
        "“我愛你”的英語是“I love you”",
        "2.5平方電線",
        "共465篇，約315萬字",
        "2002年的第一場雪，下在了2003年",
        "速度是10km/h",
        "現在是北京時間2025年01月11日 20:00",
        "他這條褲子是2012年買的，花了200塊錢",
        "電話：135-4567-8900",
        "1鍵3連",
        "他這條影片點贊3000+，評論1000+，收藏500+",
        "這是1024元的手機，你要嗎？",
        "受不liao3你了",
        "“衣裳”不讀衣chang2，而是讀衣shang5",
        "最zhong4要的是：不要chong2蹈覆轍",
        "不zuo1死就不會死",
        "See you at 8:00 AM",
        "8:00 AM 開會",
        "Couting down 3, 2, 1, go!",
        "數到3就開始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "5G網路是4G網路的升級版，2G網路是3G網路的前身",
        "蘋果於2030/1/2釋出新 iPhone 2X 系列手機，最低售價僅 ¥12999",
        "這酒...裡...有毒...",
        # 異常case
        "只有,,,才是最好的",
        "babala2是什麼？",  # babala二是什麼?
        "用beta1測試",  # 用beta一測試
        "have you ever been to beta2?",  # have you ever been to beta two?
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "今天是個好日子 it's a good day",  # 今天是個好日子 it is a good day
        # 人名
        "約瑟夫·高登-萊維特（Joseph Gordon-Levitt is an American actor）",
        "蒂莫西·唐納德·庫克（英文名：Timothy Donald Cook），通稱蒂姆·庫克（Tim Cook），美國商業經理、工業工程師和工業開發商，現任蘋果公司執行長。",
        # 長句子
        "《盜夢空間》是由美國華納兄弟影片公司出品的電影，由克里斯托弗·諾蘭執導並編劇，萊昂納多·迪卡普里奧、瑪麗昂·歌迪亞、約瑟夫·高登-萊維特、艾利奧特·佩吉、湯姆·哈迪等聯袂主演，2010年7月16日在美國上映，2010年9月1日在中國內地上映，2020年8月28日在中國內地重映。影片劇情遊走於夢境與現實之間，被定義為“發生在意識結構內的當代動作科幻片”，講述了由萊昂納多·迪卡普里奧扮演的造夢師，帶領特工團隊進入他人夢境，從他人的潛意識中盜取機密，並重塑他人夢境的故事。",
        "清晨拉開窗簾，陽光灑在窗臺的Bloomixy花藝禮盒上——薰衣草香薰蠟燭喚醒嗅覺，永生花束折射出晨露般光澤。設計師將“自然綻放美學”融入每個細節：手工陶瓷花瓶可作首飾收納，香薰精油含依蘭依蘭舒緩配方。限量款附贈《365天插花靈感手冊》，讓每個平凡日子都有花開儀式感。\n宴會廳燈光暗下的剎那，Glimmeria星月系列耳墜開始發光——瑞士冷琺琅工藝讓藍寶石如銀河流動，鈦合金骨架僅3.2g無負重感。設計師秘密：內建微型重力感應器，隨步伐產生0.01mm振幅，打造“行走的星光”。七夕限定禮盒含星座定製銘牌，讓愛意如星辰永恆閃耀。",
        "電影1：“黑暗騎士”（演員：克里斯蒂安·貝爾、希斯·萊傑；導演：克里斯托弗·諾蘭）；電影2：“盜夢空間”（演員：萊昂納多·迪卡普里奧；導演：克里斯托弗·諾蘭）；電影3：“鋼琴家”（演員：艾德里安·布洛迪；導演：羅曼·波蘭斯基）；電影4：“泰坦尼克號”（演員：萊昂納多·迪卡普里奧；導演：詹姆斯·卡梅隆）；電影5：“阿凡達”（演員：薩姆·沃辛頓；導演：詹姆斯·卡梅隆）；電影6：“南方公園：大電影”（演員：馬特·斯通、托馬斯·艾恩格瑞；導演：特雷·帕克）",
    ]
    # 測試分詞器
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model",
        normalizer=text_normalizer,
    )

    codes = tokenizer.batch_encode(
        cases,
        out_type=int,
    )

    print(f"vocab_size: {tokenizer.vocab_size}")
    # print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token: {tokenizer.unk_token}, unk_token_id: {tokenizer.unk_token_id}")
    # 測試拼音 (8474-10201)
    for id in range(8474, 10201):
        pinyin = tokenizer.convert_ids_to_tokens(id)
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, pinyin, re.IGNORECASE) is None:
            print(f"{pinyin} should be matched")
    for badcase in [
        "beta1", "better1", "voice2", "bala2", "babala2", "hunger2"
    ]:
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, badcase, re.IGNORECASE) is not None:
            print(f"{badcase} should not be matched!")
    # 不應該有 unk_token_id
    for t in set([*TextTokenizer.punctuation_marks_tokens, ",", "▁,", "-", "▁..."]):
        tokens = tokenizer.convert_tokens_to_ids(t)
        if tokenizer.unk_token_id in tokens:
            print(f"Warning: {t} is unknown token")
        print(f"`{t}`", "->", tokens, "->", tokenizer.convert_ids_to_tokens(tokens))
    for ch in set(tokenizer.normalizer.zh_char_rep_map.values()):
        # 測試 normalize後的字元能被分詞器識別
        print(f"`{ch}`", "->", tokenizer.sp_model.Encode(ch, out_type=str))
        print(f"` {ch}`", "->", tokenizer.sp_model.Encode(f" {ch}", out_type=str))
    max_tokens_per_sentence=120
    for i in range(len(cases)):
        print(f"原始文字: {cases[i]}")
        print(f"Normalized: {text_normalizer.normalize(cases[i])}")
        tokens = tokenizer.tokenize(cases[i])
        print("Tokenzied: ", ", ".join([f"`{t}`" for t in tokens]))
        sentences = tokenizer.split_sentences(tokens, max_tokens_per_sentence=max_tokens_per_sentence)
        print("Splitted sentences count:", len(sentences))
        if len(sentences) > 1:
            for j in range(len(sentences)):
                print(f"  {j}, count:", len(sentences[j]), ", tokens:", "".join(sentences[j]))
                if len(sentences[j]) > max_tokens_per_sentence:
                    print(f"Warning: sentence {j} is too long, length: {len(sentences[j])}")
        #print(f"Token IDs (first 10): {codes[i][:10]}")
        if tokenizer.unk_token in codes[i]:
            print(f"Warning: `{cases[i]}` contains UNKNOWN token")
        print(f"Decoded: {tokenizer.decode(codes[i], do_lower_case=True)}")
        print("-" * 50)
