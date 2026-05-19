from pupil_track.gui import build_dark_style, gui_scale_for_screen, scaled_px, window_size_for_screen


def test_gui_scale_preserves_4k_and_shrinks_1440p():
    assert gui_scale_for_screen(3840, 2160) == 1.0
    assert gui_scale_for_screen(2560, 1440) == 2 / 3


def test_gui_scale_clamps_small_screens_to_readable_minimum():
    assert gui_scale_for_screen(1920, 1080) == 0.65
    assert scaled_px(30, 0.65) == 20


def test_dark_style_uses_scaled_font_sizes():
    style = build_dark_style(2 / 3)

    assert "font-size: 20px" in style
    assert "QLabel#title { font-size: 28px" in style
    assert "QLabel#info  { color: #808080; font-size: 22px" in style


def test_window_size_stays_within_screen_work_area():
    assert window_size_for_screen(3840, 2160, 1.0) == (1800, 1500)

    width, height = window_size_for_screen(2560, 1440, 2 / 3)
    assert width == 1200
    assert height == 1000
    assert width <= int(2560 * 0.9)
    assert height <= int(1440 * 0.9)

    width, height = window_size_for_screen(1920, 1080, 0.65)
    assert width <= int(1920 * 0.9)
    assert height <= int(1080 * 0.9)
