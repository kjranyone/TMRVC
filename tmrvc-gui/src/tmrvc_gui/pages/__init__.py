"""GUI pages for the TMRVC Research Studio."""

from tmrvc_gui.pages.admin_dashboard import AdminDashboardPage
from tmrvc_gui.pages.codec_train import CodecTrainPage
from tmrvc_gui.pages.curation import CurationPage
from tmrvc_gui.pages.data_prep import DataPrepPage
from tmrvc_gui.pages.enrollment import EnrollmentPage
from tmrvc_gui.pages.evaluation import EvaluationPage
from tmrvc_gui.pages.onnx_export import OnnxExportPage
from tmrvc_gui.pages.realtime_demo import RealtimeDemoPage
from tmrvc_gui.pages.remote_client import RemoteClientPage
from tmrvc_gui.pages.script import ScriptPage
from tmrvc_gui.pages.server import ServerPage
from tmrvc_gui.pages.style_editor import StyleEditorPage
from tmrvc_gui.pages.token_train import TokenTrainPage
from tmrvc_gui.pages.tts import TTSPage

__all__ = [
    "AdminDashboardPage",
    "CodecTrainPage",
    "CurationPage",
    "DataPrepPage",
    "EnrollmentPage",
    "EvaluationPage",
    "OnnxExportPage",
    "RealtimeDemoPage",
    "RemoteClientPage",
    "ScriptPage",
    "ServerPage",
    "StyleEditorPage",
    "TokenTrainPage",
    "TTSPage",
]
