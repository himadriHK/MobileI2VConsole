"""
Regression test: TurboVAEDDecoder3d instantiates successfully with default inject_noise.

The default inject_noise tuple was fixed from 4 elements to 5 elements,
matching the number of up-blocks (block_out_channels has 4 entries,
so indices 0..4 are needed: inject_noise[0] for mid block and inject_noise[1..4] for up blocks).
"""

import torch.nn as nn
from models.turbo_vaed_model import TurboVAEDDecoder3d


def test_turbo_vaed_decoder_default_inject_noise_succeeds():
    """
    GIVEN the default inject_noise tuple has 5 elements (fixed)
    WHEN TurboVAEDDecoder3d is instantiated with all default arguments
    THEN no IndexError should be raised and the decoder should have valid structure.
    """
    # Arrange / Act
    decoder = TurboVAEDDecoder3d()

    # Assert — instantiation succeeds, decoder is a valid nn.Module with up_blocks
    assert isinstance(decoder, nn.Module)
    assert hasattr(decoder, "up_blocks")
    assert len(decoder.up_blocks) > 0
