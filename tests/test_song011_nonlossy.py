import unittest

from tools import rebuild_song011_nonlossy as builder
from tools.rebuild_song011_nonlossy import (
    Interval,
    build_transcription,
    detect_density_issues,
    map_empty_phones,
    parse_textgrid_text,
    validate_partition,
)


class Song011NonlossyTests(unittest.TestCase):
    def test_textgrid_parser_keeps_empty_phone_intervals(self):
        text = '''
        File type = "ooTextFile"
        Object class = "TextGrid"
        xmin = 0
        xmax = 0.5
        tiers? <exists>
        size = 1
        item []:
            item [1]:
                class = "IntervalTier"
                name = "phones"
                xmin = 0
                xmax = 0.5
                intervals: size = 3
                intervals [1]:
                    xmin = 0
                    xmax = 0.2
                    text = "a"
                intervals [2]:
                    xmin = 0.2
                    xmax = 0.4
                    text = ""
                intervals [3]:
                    xmin = 0.4
                    xmax = 0.5
                    text = "i"
        '''

        tiers = parse_textgrid_text(text)

        self.assertEqual([item.label for item in tiers["phones"]], ["a", "", "i"])
        self.assertEqual((tiers["phones"][1].start, tiers["phones"][1].end), (0.2, 0.4))

    def test_empty_phone_maps_to_sp_without_extending_lexical_phone(self):
        raw = [
            Interval(0.0, 0.2, "a"),
            Interval(0.2, 0.4, "", raw_label=""),
            Interval(0.4, 0.5, "i"),
        ]

        mapped = map_empty_phones(raw)
        sequence, durations = build_transcription(mapped)

        self.assertEqual(sequence, ["a", "SP", "i"])
        self.assertEqual(durations, [0.2, 0.2, 0.1])
        self.assertEqual((mapped[0].start, mapped[0].end), (0.0, 0.2))
        self.assertEqual((mapped[2].start, mapped[2].end), (0.4, 0.5))
        self.assertEqual(validate_partition(mapped, 0.0, 0.5), [])

    def test_generic47_alias_is_normalized_without_boundary_change(self):
        mapped = map_empty_phones([Interval(0.0, 0.1, "ɟ")])

        self.assertEqual(mapped[0].label, "ɡ")
        self.assertEqual((mapped[0].start, mapped[0].end), (0.0, 0.1))

    def test_l006_density_is_blocking_signal(self):
        phones = [Interval(index * 0.02, (index + 1) * 0.02, "a") for index in range(16)]

        issues = detect_density_issues(phones, segment_start=0.0, segment_end=0.34)

        self.assertTrue(any(issue["kind"] == "phone_density" for issue in issues))
        self.assertTrue(any(issue["kind"] == "short_segment_high_density" for issue in issues))

    def test_silent_phone_f0_is_zeroed_without_changing_voiced_frames(self):
        values = [100.0] * 10
        self.assertTrue(hasattr(builder, "zero_f0_for_silent_regions"))

        masked = builder.zero_f0_for_silent_regions(
            values,
            phones=["a", "SP", "i"],
            phone_durations=[0.03, 0.04, 0.03],
        )

        self.assertEqual(values, [100.0] * 10)
        self.assertEqual(masked, [100.0] * 3 + [0.0] * 4 + [100.0] * 3)

    def test_rest_note_f0_is_zeroed_even_when_phone_is_voiced(self):
        self.assertTrue(hasattr(builder, "zero_f0_for_silent_regions"))
        masked = builder.zero_f0_for_silent_regions(
            [200.0] * 10,
            phones=["a", "i"],
            phone_durations=[0.05, 0.05],
            note_sequence=["C4", "rest"],
            note_durations=[0.04, 0.06],
        )

        self.assertEqual(masked, [200.0] * 4 + [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
