"""Internationalization module for TMRVC GUI.

Supports Japanese (ja), English (en), and Chinese (zh).
Uses JavaScript-based translation for dynamic language switching.
"""

from __future__ import annotations

import json
from pathlib import Path

_LOCALES_DIR = Path(__file__).parent / "locales"
_current_lang = "ja"
_locales_cache: dict[str, dict] = {}


def _load_locale(lang: str) -> dict:
    """Load locale JSON file."""
    if lang in _locales_cache:
        return _locales_cache[lang]

    path = _LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _locales_cache[lang] = data
    return data


def set_language(lang: str) -> None:
    """Set current language."""
    global _current_lang
    _current_lang = lang


def get_tooltip(key: str, lang: str | None = None) -> str:
    """Get tooltip text for a key."""
    target_lang = lang or _current_lang
    locale = _load_locale(target_lang)
    tooltips = locale.get("tooltips", {})
    return tooltips.get(key, "")


def t(key: str, lang: str | None = None) -> str:
    """Get translated text for a key (dot notation supported)."""
    target_lang = lang or _current_lang
    locale = _load_locale(target_lang)
    parts = key.split(".")
    value = locale
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return key
    return str(value) if isinstance(value, str) else key


_LOCALES_JS = """
<script>
window.TMRVC_LOCALES = {
    ja: {
        tabs: {
            "tab-tts": "TTS合成",
            "tab-vc": "音声変換",
            "tab-curation": "キュレーション",
            "tab-casting": "キャスティング",
            "tab-eval": "A/B評価",
            "tab-datasets": "データセット",
            "tab-settings": "設定",
            "tab-admin": "管理",
            "tab-batch": "バッチ生成"
        },
        common: { submit: "実行", cancel: "キャンセル", save: "保存", load: "読込", clear: "クリア", refresh: "更新", download: "ダウンロード", upload: "アップロード", delete: "削除" }
    },
    en: {
        tabs: {
            "tab-tts": "TTS Synthesis",
            "tab-vc": "Voice Conversion",
            "tab-curation": "Curation",
            "tab-casting": "Casting",
            "tab-eval": "A/B Evaluation",
            "tab-datasets": "Datasets",
            "tab-settings": "Settings",
            "tab-admin": "Admin",
            "tab-batch": "Batch Generation"
        },
        common: { submit: "Submit", cancel: "Cancel", save: "Save", load: "Load", clear: "Clear", refresh: "Refresh", download: "Download", upload: "Upload", delete: "Delete" }
    },
    zh: {
        tabs: {
            "tab-tts": "TTS合成",
            "tab-vc": "语音转换",
            "tab-curation": "数据整理",
            "tab-casting": "角色选角",
            "tab-eval": "A/B评估",
            "tab-datasets": "数据集",
            "tab-settings": "设置",
            "tab-admin": "系统管理",
            "tab-batch": "批量生成"
        },
        common: { submit: "提交", cancel: "取消", save: "保存", load: "加载", clear: "清空", refresh: "刷新", download: "下载", upload: "上传", delete: "删除" }
    }
};

window.tmrvc_i18n = {
    currentLang: 'ja',
    
    setLang: function(lang) {
        this.currentLang = lang;
        localStorage.setItem('tmrvc_lang', lang);
        this.updateUI();
    },
    
    updateUI: function() {
        var self = this;
        var locale = window.TMRVC_LOCALES[this.currentLang];
        
        // Update tab labels by elem_id
        if (locale.tabs) {
            Object.keys(locale.tabs).forEach(function(tabId) {
                var tab = document.getElementById(tabId);
                if (tab) {
                    var button = tab.querySelector('button');
                    if (button) {
                        button.textContent = locale.tabs[tabId];
                    }
                }
            });
        }
        
        // Update elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(function(el) {
            var key = el.getAttribute('data-i18n');
            var value = self.getValue(key);
            if (value) {
                el.textContent = value;
            }
        });
    },
    
    getValue: function(key) {
        var parts = key.split('.');
        var value = window.TMRVC_LOCALES[this.currentLang];
        for (var i = 0; i < parts.length; i++) {
            if (value && value[parts[i]]) {
                value = value[parts[i]];
            } else {
                return null;
            }
        }
        return value;
    },
    
    init: function() {
        var saved = localStorage.getItem('tmrvc_lang');
        if (saved && saved !== this.currentLang) {
            this.currentLang = saved;
        }
        this.updateUI();
    }
};

// Run init after Gradio loads
setTimeout(function() {
    window.tmrvc_i18n.init();
}, 500);
</script>
"""

_HEAD_JS = """
<script>
(function() {
    var saved = localStorage.getItem('tmrvc_lang');
    window.tmrvc_current_lang = saved || 'ja';
})();
</script>
"""


def get_head_js() -> str:
    return _HEAD_JS


def get_locales_js() -> str:
    return _LOCALES_JS


available_languages = [
    {"code": "ja", "name": "日本語"},
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "中文"},
]
