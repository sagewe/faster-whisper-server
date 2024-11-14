import os


def get_language():
    return os.getenv("I18N_LANGUAGE", "zh")


class I18nText(str):
    def __new__(cls, zh: str, en: str = None):
        if en is None:
            en = zh
        if get_language() == "zh":
            value = zh
        else:
            value = en
        obj = super().__new__(cls, value)
        obj.zh = zh
        obj.en = en
        return obj

    def __reduce__(self):
        return (self.__class__, (self.zh, self.en))
