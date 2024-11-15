from faster_whisper_server.apps.transcription.i18n import I18nText

TRANSCRIPTION_HEIGHLIGHT_STYLE = dict(
    label=I18nText(
        "识别结果(绿色为当前确认结果,红色为未确认结果)",
        "Transcription (Red for current confirmed, Green for unconfirmed)",
    ),
    interactive=False,
    show_inline_category=False,
    show_legend=False,
    combine_adjacent=True,
    color_map={"confirmed": "green", "unconfirmed": "red", "info": "gray"},
)
