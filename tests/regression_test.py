from indextts.infer import IndexTTS

if __name__ == "__main__":
    prompt_wav="tests/sample_prompt.wav"
    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    # 單音訊推理測試
    text="暈 XUAN4 是 一 種 GAN3 覺"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text='大家好，我現在正在bilibili 體驗 ai 科技，說實話，來之前我絕對想不到！AI技術已經發展到這樣匪夷所思的地步了！'
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text="There is a vehicle arriving in dock number 7?"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text = "“我愛你！”的英語是“I love you!”"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text = "Joseph Gordon-Levitt is an American actor"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text = "約瑟夫·高登-萊維特是美國演員"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text = "蒂莫西·唐納德·庫克（英文名：Timothy Donald Cook），通稱蒂姆·庫克（Tim Cook），現任蘋果公司執行長。"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="outputs/蒂莫西·唐納德·庫克.wav", verbose=True)
    # 並行推理測試
    text="親愛的夥伴們，大家好！每一次的努力都是為了更好的未來，要善於從失敗中汲取經驗，讓我們一起勇敢前行,邁向更加美好的明天！"
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text="The weather is really nice today, perfect for studying at home.Thank you!"
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text='''葉遠隨口答應一聲，一定幫忙云云。
教授看葉遠的樣子也知道，這事情多半是黃了。
誰得到這樣的東西也不會輕易貢獻出來，這是很大的一筆財富。
葉遠回來後，又自己做了幾次試驗，發現空間湖水對一些外傷也有很大的幫助。
找來一隻斷了腿的兔子，喝下空間湖水，一天時間，兔子就完全好了。
還想多做幾次試驗，可是身邊沒有試驗的物件，就先放到一邊，瞭解空間湖水可以飲用，而且對人有利，這些就足夠了。
感謝您的收聽，下期再見！
    '''.replace("\n", "")
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    # 長文字推理測試
    text = """《盜夢空間》是由美國華納兄弟影片公司出品的電影，由克里斯托弗·諾蘭執導並編劇，
萊昂納多·迪卡普里奧、瑪麗昂·歌迪亞、約瑟夫·高登-萊維特、艾利奧特·佩吉、湯姆·哈迪等聯袂主演，
2010年7月16日在美國上映，2010年9月1日在中國內地上映，2020年8月28日在中國內地重映。
影片劇情遊走於夢境與現實之間，被定義為“發生在意識結構內的當代動作科幻片”，
講述了由萊昂納多·迪卡普里奧扮演的造夢師，帶領特工團隊進入他人夢境，從他人的潛意識中盜取機密，並重塑他人夢境的故事。
""".replace("\n", "")
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
