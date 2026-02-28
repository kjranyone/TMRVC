"""Tests for streaming_codec module."""

import pytest
import torch

from tmrvc_train.models.streaming_codec import (
    CodecConfig,
    StreamingCodec,
    StreamingCodecEncoder,
    StreamingCodecDecoder,
    ResidualVectorQuantizer,
    CausalConv1d,
    ENCODER_STATE_DIM,
    ENCODER_STATE_FRAMES,
    DECODER_STATE_DIM,
    DECODER_STATE_FRAMES,
)


class TestCodecConfig:
    def test_default_config(self):
        config = CodecConfig()
        assert config.frame_size == 480
        assert config.sample_rate == 24000
        assert config.n_codebooks == 4
        assert config.codebook_size == 1024
        assert config.codebook_dim == 128

    def test_frame_rate(self):
        config = CodecConfig()
        assert config.frame_rate == 50

    def test_token_rate(self):
        config = CodecConfig()
        assert config.token_rate == 200


class TestCausalConv1d:
    def test_forward_shape(self):
        conv = CausalConv1d(1, 32, kernel_size=7)
        x = torch.randn(1, 1, 480)
        out, state = conv(x, None)
        assert out.shape == (1, 32, 480)

    def test_state_shape(self):
        conv = CausalConv1d(1, 32, kernel_size=7)
        x = torch.randn(1, 1, 480)
        _, state = conv(x, None)
        assert state.shape == (1, 1, 6)


class TestResidualVectorQuantizer:
    def test_forward_shape(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=4, codebook_size=1024, codebook_dim=128
        )
        x = torch.randn(1, 128, 480)
        quantized, indices, commit_loss = rvq(x)
        assert quantized.shape == (1, 128, 480)
        assert indices.shape == (1, 4, 480)

    def test_decode_shape(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=4, codebook_size=1024, codebook_dim=128
        )
        indices = torch.randint(0, 1024, (1, 4, 480))
        quantized = rvq.decode(indices)
        assert quantized.shape == (1, 128, 480)


class TestStreamingCodecEncoder:
    def test_forward_shape(self):
        config = CodecConfig()
        encoder = StreamingCodecEncoder(config)
        x = torch.randn(1, 1, 480)
        out, state = encoder(x, None)
        assert out.shape[0] == 1
        assert out.shape[2] == 480
        assert state.shape == (1, ENCODER_STATE_DIM, ENCODER_STATE_FRAMES)


class TestStreamingCodecDecoder:
    def test_forward_shape(self):
        config = CodecConfig()
        decoder = StreamingCodecDecoder(config)
        x = torch.randn(1, config.codebook_dim, 480)
        out, state = decoder(x, None)
        assert out.shape == (1, 1, 480)
        assert state.shape == (1, DECODER_STATE_DIM, DECODER_STATE_FRAMES)


class TestStreamingCodec:
    def test_encode_shape(self):
        codec = StreamingCodec()
        x = torch.randn(1, 1, 480)
        indices, quantized, state = codec.encode(x, None)
        assert indices.shape == (1, 4, 480)
        assert quantized.shape == (1, 128, 480)

    def test_decode_shape(self):
        codec = StreamingCodec()
        indices = torch.randint(0, 1024, (1, 4, 480))
        audio, state = codec.decode(indices, None)
        assert audio.shape == (1, 1, 480)

    def test_full_roundtrip(self):
        codec = StreamingCodec()
        x = torch.randn(1, 1, 480)
        enc_state = codec.encoder.init_state(1, torch.device("cpu"))
        dec_state = codec.decoder.init_state(1, torch.device("cpu"))

        audio_out, indices, commit_loss, enc_state_out, dec_state_out = codec(
            x, enc_state, dec_state
        )

        assert audio_out.shape == x.shape
        assert indices.shape == (1, 4, 480)
