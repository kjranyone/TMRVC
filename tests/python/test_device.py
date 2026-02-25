"""Tests for tmrvc_core.device module."""

from __future__ import annotations

from unittest.mock import patch

import torch

from tmrvc_core.device import get_device, pin_memory_for_device


class TestGetDevice:
    def test_cpu_explicit(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_cpu_uppercase(self):
        device = get_device("CPU")
        assert device == torch.device("cpu")

    @patch("tmrvc_core.device._xpu_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _mock_cuda, _mock_xpu):
        device = get_device("auto")
        assert device == torch.device("cpu")

    @patch("tmrvc_core.device._xpu_available", return_value=True)
    def test_auto_selects_xpu(self, _mock_xpu):
        device = get_device("auto")
        assert device == torch.device("xpu")

    @patch("tmrvc_core.device._xpu_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda(self, _mock_cuda, _mock_xpu):
        device = get_device("auto")
        assert device == torch.device("cuda")

    @patch("tmrvc_core.device._xpu_available", return_value=False)
    def test_xpu_requested_but_unavailable(self, _mock_xpu):
        device = get_device("xpu")
        assert device == torch.device("cpu")

    @patch("tmrvc_core.device._xpu_available", return_value=True)
    def test_xpu_requested_and_available(self, _mock_xpu):
        device = get_device("xpu")
        assert device == torch.device("xpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_requested_but_unavailable(self, _mock_cuda):
        device = get_device("cuda")
        assert device == torch.device("cpu")

    def test_unknown_device_returns_cpu(self):
        device = get_device("tpu")
        assert device == torch.device("cpu")


class TestPinMemory:
    def test_cuda_gets_pin_memory(self):
        assert pin_memory_for_device(torch.device("cuda")) is True

    def test_cuda_string(self):
        assert pin_memory_for_device("cuda:0") is True

    def test_cpu_no_pin(self):
        assert pin_memory_for_device(torch.device("cpu")) is False

    def test_xpu_no_pin(self):
        assert pin_memory_for_device(torch.device("xpu")) is False
