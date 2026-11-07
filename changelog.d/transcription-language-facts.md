### Added

- Transcription language facts: `generate(..., output={"task": "transcription"})` records `language` (what the engine was told; `None` = it detected the language) and `detected_language` (what it reported) in the transcript's metadata, through the audio facade's `transcribe_detailed` (`AudioCapability` may implement it — AbstractVoice does; a backend with `transcribe` only reports neither). `POST /v1/audio/transcriptions` on a local engine answers `{text, language, detected_language}`.
