"""MIDI tempo map、音符和复音异常解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MidiResult:
    notes: list[dict[str, Any]]
    tempo_events: list[dict[str, Any]]
    issues: list[dict[str, Any]]


def midi_note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def parse_midi(path: Path) -> MidiResult:
    try:
        import mido
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"缺少 mido，无法解析 MIDI: {exc}") from exc

    midi = mido.MidiFile(str(path))
    all_notes: list[dict[str, Any]] = []
    tempo_events: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        tempo = 500000
        seconds = 0.0
        active: dict[int, list[float]] = {}
        timed_messages: list[tuple[float, int, Any]] = []
        for order, message in enumerate(track):
            seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
            timed_messages.append((seconds, order, message))
            if message.type == "set_tempo":
                tempo = message.tempo
        index = 0
        while index < len(timed_messages):
            timestamp = timed_messages[index][0]
            group: list[tuple[float, int, Any]] = []
            while index < len(timed_messages) and abs(timed_messages[index][0] - timestamp) <= 1e-9:
                group.append(timed_messages[index])
                index += 1
            # 许多导出器在同一 tick 先写下一音符的 note_on，再写上一音符的 note_off；
            # 先处理结束事件，避免把首尾相接误报成复音，同时保留真正重叠的检测。
            group.sort(key=lambda item: (0 if item[2].type == "note_off" or (item[2].type == "note_on" and item[2].velocity == 0) else 1, item[1]))
            for _, _, message in group:
                if message.type == "set_tempo":
                    tempo = message.tempo
                    tempo_events.append({"track": track_index, "time": timestamp, "bpm": 60000000 / tempo})
                elif message.type == "note_on" and message.velocity > 0:
                    if any(active.values()):
                        issues.append(
                            {
                                "type": "OVERLAPPING_NOTES",
                                "segment_id": f"track-{track_index}",
                                "start_sec": f"{timestamp:.6f}",
                                "message": "MIDI 存在复音或重叠音符，需要选择旋律轨",
                            }
                        )
                    active.setdefault(message.note, []).append(timestamp)
                elif message.type in {"note_off", "note_on"} and (message.type == "note_off" or message.velocity == 0):
                    starts = active.get(message.note, [])
                    if not starts:
                        issues.append({"type": "NOTE_OFF_WITHOUT_ON", "segment_id": f"track-{track_index}", "message": "MIDI note_off 无对应 note_on"})
                        continue
                    start = starts.pop(0)
                    if timestamp <= start:
                        issues.append({"type": "NON_POSITIVE_NOTE_DURATION", "segment_id": f"track-{track_index}", "message": "MIDI 音符时长非正"})
                    all_notes.append(
                        {
                            "track": track_index,
                            "pitch": message.note,
                            "note": midi_note_name(message.note),
                            "start": start,
                            "end": timestamp,
                            "duration": timestamp - start,
                        }
                    )
        for pitch, starts in active.items():
            for start in starts:
                issues.append({"type": "UNRELEASED_NOTE", "segment_id": f"track-{track_index}", "start_sec": f"{start:.6f}", "message": "MIDI 音符没有结束事件"})
    all_notes.sort(key=lambda item: (item["start"], item["pitch"], item["track"]))
    track_ids = {item["track"] for item in all_notes}
    if len(track_ids) > 1:
        issues.append({"type": "MULTI_TRACK_SCORE", "message": "MIDI 有多个含音符轨道，需要人工选择旋律轨"})
    return MidiResult(notes=all_notes, tempo_events=tempo_events, issues=issues)
