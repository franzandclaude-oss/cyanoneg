"""Colour-space handling: the tests guarding against silent tonal errors."""

import io
import struct

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageCms

from cyanoneg import imageio as cio

SIXTEEN_BIT_STEP = 1.0 / 65535.0

# Adobe RGB (1998) colorants, already adapted to the D50 PCS as ICC requires.
ADOBE_COLORANTS = (
    (0.60974121, 0.31111145, 0.01947021),
    (0.20527649, 0.62567139, 0.06086731),
    (0.14918518, 0.06321716, 0.74456787),
)


def _s15f16(x: float) -> bytes:
    return struct.pack(">i", round(x * 65536.0))


def xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + bytes(4) + _s15f16(x) + _s15f16(y) + _s15f16(z)


def curv_gamma(gamma: float) -> bytes:
    return b"curv" + bytes(4) + struct.pack(">I", 1) + struct.pack(">H", round(gamma * 256))


def curv_table(values) -> bytes:
    v = np.asarray(values, dtype=np.float64)
    quantised = np.rint(np.clip(v, 0, 1) * 65535).astype(">u2")
    return b"curv" + bytes(4) + struct.pack(">I", len(v)) + quantised.tobytes()


def fake_profile(tags: dict[str, bytes]) -> bytes:
    """A blob with a valid ICC tag table and nothing else.

    Enough for this module's own parser, which is the point: the grey and refusal paths
    can be exercised without hand-forging a profile littleCMS would accept.
    """
    header = bytes(128)
    table = struct.pack(">I", len(tags))
    offset = 128 + 4 + 12 * len(tags)
    body = b""
    for name, data in tags.items():
        table += struct.pack(">4sII", name.encode("ascii"), offset + len(body), len(data))
        body += data
    return header + table + body


def mluc(text: str) -> bytes:
    """An ICC v4 multi-localised description tag holding one en-US string."""
    body = text.encode("utf-16-be")
    return (
        b"mluc"
        + bytes(4)
        + struct.pack(">II", 1, 12)
        + struct.pack(">4sII", b"enUS", len(body), 28)
        + body
    )


def rebuild_profile(base: bytes, replacements: dict[str, bytes]) -> bytes:
    """Rewrite a profile with some tags replaced, recomputing the tag table.

    Editing tags in place would be simpler but only works when the new tag is exactly the
    old size, which rules out changing the description — and a fixture that claims to be
    sRGB while carrying foreign primaries would be tested down the wrong code path.
    """
    tags = dict(cio._icc_tags(base))
    tags.update(replacements)
    table, body = struct.pack(">I", len(tags)), b""
    start = 128 + 4 + 12 * len(tags)
    for name, data in tags.items():
        table += struct.pack(">4sII", name.encode("ascii"), start + len(body), len(data))
        body += data + bytes(-len(data) % 4)  # tag data is 4-byte aligned
    icc = bytearray(base[:128] + table + body)
    struct.pack_into(">I", icc, 0, len(icc))  # the header carries the total size
    return bytes(icc)


def wide_gamut_profile(colorants=ADOBE_COLORANTS, name="Adobe RGB (1998)") -> bytes:
    """A real, littleCMS-readable profile that is genuinely not sRGB.

    Built from Pillow's sRGB profile so the header and curves stay valid, with the
    colorants and the description replaced.
    """
    xyz = {f"{c}XYZ": xyz_tag(*colorants[i]) for i, c in enumerate("rgb")}
    return rebuild_profile(cio.srgb_icc(), {**xyz, "desc": mluc(name)})


class TestTransferCurves:
    @pytest.mark.parametrize("space", cio.SPACES)
    def test_round_trip_well_below_16bit_precision(self, space):
        x = np.linspace(0.0, 1.0, 100_001, dtype=np.float32)
        back = cio.from_linear(cio.to_linear(x, space), space)
        assert np.abs(back - x).max() < SIXTEEN_BIT_STEP / 10

    def test_srgb_is_not_gamma22(self):
        """The two curves differ most in the shadows — conflating them is the classic error."""
        x = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
        divergence = np.abs(cio.to_linear(x, "srgb") - cio.to_linear(x, "gamma22")).max()
        assert divergence > 0.005

    def test_srgb_linear_segment(self):
        """Below the breakpoint sRGB is linear (slope 1/12.92), not a power curve."""
        x = np.array([0.001, 0.02, 0.04], dtype=np.float32)
        assert np.allclose(cio.to_linear(x, "srgb"), x / 12.92, atol=1e-7)

    def test_unknown_space_rejected(self):
        with pytest.raises(ValueError, match="unknown colour space"):
            cio.to_linear(np.zeros(4), "adobe98")

    def test_convert_space_identity_when_same(self):
        x = np.random.default_rng(0).random(100).astype(np.float32)
        assert np.array_equal(cio.convert_space(x, "srgb", "srgb"), x)


class TestLoadSave:
    def test_chart_loads_as_srgb_300ppi_16bit(self, chart_path):
        img = cio.load_image(chart_path)
        assert img.space == "srgb"
        assert img.ppi == 300.0
        assert img.bit_depth == 16
        assert img.data.dtype == np.float32
        assert img.data.shape == (1507, 1507, 3)

    def test_untagged_file_raises_without_explicit_space(self, tmp_path):
        """Guessing a colour space is forbidden — the file must be tagged or told."""
        import tifffile

        path = tmp_path / "untagged.tif"
        tifffile.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint16))
        with pytest.raises(cio.ColourSpaceError, match="has no ICC profile"):
            cio.load_image(path)
        # ...but an explicit override is honoured.
        assert cio.load_image(path, space="gamma22").space == "gamma22"

    def test_save_reload_round_trip(self, tmp_path, ramp_image):
        path = cio.save_tiff(tmp_path / "rt.tif", ramp_image)
        back = cio.load_image(path)
        assert back.space == "srgb"
        assert back.ppi == 300.0
        # Arbitrary float data survives to within half a 16-bit quantisation step…
        assert np.abs(back.data - ramp_image.data).max() <= 0.5 / 65535.0
        # …and already-quantised data round-trips bit-exact.
        again = cio.load_image(cio.save_tiff(tmp_path / "rt2.tif", back))
        assert np.array_equal(again.data, back.data)

    def test_save_bare_array_requires_space(self, tmp_path):
        with pytest.raises(cio.ColourSpaceError):
            cio.save_tiff(tmp_path / "x.tif", np.zeros((4, 4, 3), dtype=np.float32))


class TestCurveParsing:
    """The ICC curve types, checked against maths that does not come from this module."""

    def test_curv_single_gamma(self):
        curve = cio._parse_curve(curv_gamma(2.19921875))  # Adobe RGB's actual gamma
        x = np.linspace(0, 1, 257)
        assert np.abs(curve(x) - x**2.19921875).max() < 1e-9

    def test_curv_identity(self):
        curve = cio._parse_curve(b"curv" + bytes(4) + struct.pack(">I", 0))
        x = np.linspace(0, 1, 33)
        assert np.array_equal(curve(x), x)

    def test_curv_table_interpolates(self):
        grid = np.linspace(0, 1, 64)
        curve = cio._parse_curve(curv_table(grid**2))
        x = np.linspace(0, 1, 257)
        assert np.abs(curve(x) - x**2).max() < 0.002  # table resolution, not a maths error

    def test_para_matches_our_own_srgb_curve(self):
        """Pillow's sRGB profile stores a type-3 parametric curve.

        Decoding it must land on the piecewise sRGB function this module implements by
        hand. Two independent statements of the same curve, so a mistake in either the
        parser or the analytic version shows up here.
        """
        tags = cio._icc_tags(cio.srgb_icc())
        curve = cio._parse_curve(tags["rTRC"])
        x = np.linspace(0, 1, 4097, dtype=np.float32)
        # The profile stores its parameters as s15Fixed16 — gamma comes back as 2.3999939,
        # not 2.4 — so exact agreement is not available. This is ~0.001 of an 8-bit code.
        assert np.abs(curve(x) - cio.to_linear(x, "srgb")).max() < 1e-5

    def test_unsupported_curve_type_refused(self):
        with pytest.raises(ValueError, match="unsupported curve tag type"):
            cio._parse_curve(b"mAB " + bytes(20))


class TestEmbeddedProfiles:
    """Reading a profile the file declares is not guessing — but it must be *right*."""

    def test_srgb_converts_to_itself(self):
        """The whole path — parse, linearise, to XYZ, back, re-encode — as an identity.

        Anything wrong in the matrix, its inverse, or the curve shows up as drift here,
        with no oracle required.
        """
        rng = np.random.default_rng(0)
        data = rng.random((32, 32, 3)).astype(np.float32)
        out = cio.icc_to_srgb(data, cio.srgb_icc())
        assert np.abs(out - data).max() < 1.0 / 65535.0

    def test_matches_littlecms_on_foreign_primaries(self):
        """littleCMS as the oracle, on a profile that is genuinely not sRGB.

        Agreement well inside one 8-bit code value means the numpy path can be trusted for
        the 16-bit scans PIL cannot transform at all.
        """
        icc = wide_gamut_profile()
        lattice = (np.indices((16, 16, 16)).reshape(3, -1).T / 15.0).astype(np.float32)

        as_bytes = PILImage.fromarray(
            (lattice.reshape(1, -1, 3) * 255).round().astype(np.uint8), "RGB"
        )
        truth = np.asarray(
            ImageCms.profileToProfile(
                as_bytes,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            ),
            dtype=np.float32,
        ).reshape(-1, 3) / 255.0

        mine = cio.icc_to_srgb(lattice.reshape(1, -1, 3), icc).reshape(-1, 3)
        assert np.abs(mine - truth).max() < 1.0 / 255.0

    def test_a_wider_gamut_actually_changes_the_pixels(self):
        """Guards the oracle test above: if the patched profile were still sRGB, both
        implementations would agree on doing nothing and prove nothing."""
        data = np.full((4, 4, 3), 0.5, dtype=np.float32)
        data[..., 1] = 0.9  # saturated green, where the primaries differ most
        out = cio.icc_to_srgb(data, wide_gamut_profile())
        assert np.abs(out - data).max() > 0.02

    def test_grey_profile_lands_on_the_neutral_axis(self):
        """A grey profile has only a neutral axis, and the conversion must keep it there.

        The expectation is exact rather than approximate: the sRGB colorants sum to the
        D50 white by construction, so grey in must come out as equal channels carrying
        exactly the tone curve.
        """
        icc = fake_profile({"kTRC": curv_gamma(2.2), "wtpt": xyz_tag(0.9642, 1.0, 0.8249)})
        grey = np.array([[0.25, 0.5, 0.75]], dtype=np.float32)
        out = cio.icc_to_srgb(np.stack([grey] * 3, axis=-1), icc)
        # A curv gamma is u8Fixed8, so "2.2" is stored as 563/256 and the expectation has
        # to use the value the format can actually hold.
        expected = cio.from_linear(grey ** (563 / 256), "srgb")
        assert np.abs(out - expected[..., None]).max() < 1e-4
        # Not bit-exact neutral: the fixture's D50 is rounded to four places while sRGB's
        # colorants carry the full value, so the axis tilts by ~1e-5 — 0.003 of an 8-bit code.
        assert np.ptp(out, axis=-1).max() < 1e-4, "channels must stay equal"

    def test_grey_profile_on_mono_data_stays_mono(self):
        icc = fake_profile({"kTRC": curv_gamma(2.2), "wtpt": xyz_tag(0.9642, 1.0, 0.8249)})
        out = cio.icc_to_srgb(np.full((4, 4), 0.5, dtype=np.float32), icc)
        assert out.shape == (4, 4)

    def test_non_matrix_profile_is_declined_not_guessed(self):
        """A LUT-based profile cannot be reduced to a matrix, and must not be faked."""
        lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
        assert cio._matrix_shaper(lab) is None
        assert cio.icc_to_srgb(np.zeros((2, 2, 3), dtype=np.float32), lab) is None

    def test_garbage_profile_declined(self):
        assert cio.icc_to_srgb(np.zeros((2, 2, 3), dtype=np.float32), b"not a profile") is None


class TestLoadWithConversion:
    @pytest.fixture
    def adobe_tiff(self, tmp_path):
        import tifffile

        data = np.linspace(0, 1, 24 * 3, dtype=np.float32).reshape(8, 3, 3)
        icc = wide_gamut_profile()
        path = tmp_path / "wide.tif"
        tifffile.imwrite(
            path,
            np.rint(data * 65535).astype(np.uint16),
            photometric="rgb",
            extratags=[(34675, 7, len(icc), icc, True)],
        )
        return path, data

    def test_converts_and_says_so(self, adobe_tiff):
        """A silent conversion would be exactly the implicit behaviour this module forbids."""
        path, data = adobe_tiff
        img = cio.load_image(path)
        assert img.space == "srgb"
        assert img.converted_from  # names the profile it came from
        assert np.abs(img.data - data).max() > 1.0 / 255.0  # pixels really were converted

    def test_explicit_space_overrides_and_does_not_convert(self, adobe_tiff):
        """`space=` means "trust these numbers", so nothing may be transformed underneath."""
        path, data = adobe_tiff
        img = cio.load_image(path, space="gamma22")
        assert img.space == "gamma22"
        assert img.converted_from is None
        assert np.abs(img.data - data).max() <= 0.5 / 65535.0

    def test_srgb_file_is_passed_through_untouched(self, chart_path):
        """The fast path must stay bit-exact — no round trip through XYZ for sRGB files."""
        img = cio.load_image(chart_path)
        assert img.space == "srgb" and img.converted_from is None

    def test_conversion_survives_the_whole_pipeline(self, adobe_tiff):
        """Provenance has to reach the far end, or "we can always say what we did" is false.

        Every pipeline step rebuilds the Image — some via replace(), two by constructing
        one outright — and each is a place the record can be dropped without any visible
        symptom.
        """
        from cyanoneg.pipeline import PrintSize, make_negative
        from cyanoneg.profiles import PROFILE_DIR, Profile

        path, _ = adobe_tiff
        img = cio.load_image(path)
        assert img.replace(img.data * 0.5).converted_from == img.converted_from
        negative = make_negative(img, Profile.load(PROFILE_DIR / "linear.json"), PrintSize(20, 20))
        assert negative.converted_from == img.converted_from
