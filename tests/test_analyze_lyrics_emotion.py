import unittest

from analyze_lyrics_emotion import build_parameter_schedule, parse_srt_text


class AnalyzeLyricsEmotionTests(unittest.TestCase):
    def test_parse_srt_text_keeps_japanese_line_and_timestamp(self):
        text = (
            "1\n"
            "00:00:30,780 --> 00:00:35,880\n"
            "テスト音声　サンプル歌詞\n"
            "测试音频\n"
            "\n"
            "2\n"
            "00:00:35,880 --> 00:00:41,550\n"
            "わかってる\n"
            "我知道的\n"
        )

        records = parse_srt_text(text)

        self.assertEqual([item["id"] for item in records], [1, 2])
        self.assertEqual(records[0]["text"], "テスト音声　サンプル歌詞")
        self.assertAlmostEqual(records[0]["start"], 30.780, places=3)
        self.assertAlmostEqual(records[1]["end"], 41.550, places=3)

    def test_build_parameter_schedule_clamps_rvc_controls(self):
        records = [
            {
                "id": 1,
                "text": "テスト音声",
                "start": 30.780,
                "end": 35.880,
                "duration": 5.100,
                "semantic_scores": {
                    "joy": 1.0,
                    "sadness": 0.0,
                    "anticipation": 1.0,
                    "surprise": 1.0,
                    "anger": 1.0,
                    "fear": 1.0,
                    "disgust": 1.0,
                    "trust": 1.0,
                },
                "acoustic_intensity": 1.0,
            }
        ]

        segments = build_parameter_schedule(records, groups=[("test", 1, 1)])

        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(segments[0]["index_rate"], 0.54)
        self.assertLessEqual(segments[0]["index_rate"], 0.62)
        self.assertGreaterEqual(segments[0]["rms_mix_rate"], 0.28)
        self.assertLessEqual(segments[0]["rms_mix_rate"], 0.35)
        self.assertGreaterEqual(segments[0]["protect"], 0.22)
        self.assertLessEqual(segments[0]["protect"], 0.27)

    def test_build_parameter_schedule_uses_generic_default_group(self):
        records = [
            {
                "id": 1,
                "text": "テスト",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "semantic_scores": {label: 0.05 for label in (
                    "joy", "sadness", "anticipation", "surprise",
                    "anger", "fear", "disgust", "trust",
                )},
                "acoustic_intensity": 0.5,
            },
            {
                "id": 2,
                "text": "サンプル",
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "semantic_scores": {label: 0.05 for label in (
                    "joy", "sadness", "anticipation", "surprise",
                    "anger", "fear", "disgust", "trust",
                )},
                "acoustic_intensity": 0.5,
            },
        ]

        segments = build_parameter_schedule(records)

        self.assertEqual([segment["name"] for segment in segments], ["full_song"])


if __name__ == "__main__":
    unittest.main()
