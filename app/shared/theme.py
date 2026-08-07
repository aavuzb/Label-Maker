"""App-wide visual design: fixed, explicit colors and a small type scale (not
OS palette roles or ad-hoc sizes scattered across widgets).

Relying on QPalette roles like `palette(mid)` in a stylesheet makes the
result dependent on the host desktop theme — on some systems that role
resolves to a color almost indistinguishable from the background, which is
why buttons/text could look borderless or invisible. Everything here is a
hardcoded hex value plus an explicit QPalette, so the app looks the same
(and stays legible) on every machine.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap

BG_MAIN = "#f4f5f9"
BG_SURFACE = "#ffffff"
BORDER = "#dde1ea"
BORDER_STRONG = "#b9bfcc"
ACCENT = "#6366f1"
ACCENT_HOVER = "#4f46e5"
ACCENT_TEXT = "#4338ca"
ACCENT_SOFT_BG = "#eef0ff"
TEXT_PRIMARY = "#1a2233"
TEXT_MUTED = "#657085"
TEXT_DISABLED = "#a3aab8"
RADIUS = 8
RADIUS_CARD = 12

TASK_ACCENTS = {
    "classification": "#6366f1",  # indigo
    "detection": "#0d9488",  # teal
    "segmentation": "#d97706",  # amber
}

# -- typography --------------------------------------------------------
#
# A short, deliberate type scale used everywhere instead of ad-hoc pixel
# sizes, so headings/body/captions stay visually consistent across every
# page. Font family is a fallback chain (via QFont.setFamilies / QSS
# comma-list) so the app still looks clean on whichever of these is
# actually installed, rather than silently falling back to a generic
# system serif/bitmap font.
FONT_FALLBACKS = ["Segoe UI", "Inter", "Ubuntu", "Noto Sans", "Cantarell", "Helvetica Neue", "Arial"]
FONT_FAMILY_CSS = ", ".join(f'"{name}"' for name in FONT_FALLBACKS) + ", sans-serif"

FS_DISPLAY = 30  # landing page title
FS_H1 = 20  # rarely used page-level heading
FS_H2 = 13  # panel/section header (uppercase, letter-spaced)
FS_BODY = 14  # default widget text
FS_SMALL = 13  # secondary text, buttons on dense rows
FS_CAPTION = 12  # subtitles, hints, badges


def app_font() -> QFont:
    font = QFont(FONT_FALLBACKS[0])
    font.setFamilies(FONT_FALLBACKS)
    font.setPointSize(11)
    return font


def _shade(hex_color: str, factor: float) -> str:
    """Multiplies each RGB channel by `factor` (<1 darkens, >1 lightens)."""
    r = min(255, max(0, int(int(hex_color[1:3], 16) * factor)))
    g = min(255, max(0, int(int(hex_color[3:5], 16) * factor)))
    b = min(255, max(0, int(int(hex_color[5:7], 16) * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def tint_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def readable_text_color(bg_hex: str) -> QColor:
    """Picks black or white text for contrast against an arbitrary
    background color, since class colors are user-chosen and can be light
    or dark."""
    color = QColor(bg_hex)
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return QColor("#000000") if luminance > 150 else QColor("#ffffff")


def primary_button_style(accent: str = ACCENT) -> str:
    """A solid, filled CTA button in the given accent — for the one main
    action on a page (Assign, Export…)."""
    hover = _shade(accent, 0.88)
    pressed = _shade(accent, 0.78)
    return f"""
        QPushButton {{
            background-color: {accent};
            color: #ffffff;
            border: 1px solid {accent};
            border-radius: {RADIUS}px;
            padding: 7px 18px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {hover}; border-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; border-color: {pressed}; }}
        QPushButton:disabled {{ background-color: {TEXT_DISABLED}; border-color: {TEXT_DISABLED}; color: #ffffff; }}
    """


def outline_button_style(accent: str) -> str:
    """A secondary action button — outlined in the given accent, filled on hover."""
    return f"""
        QPushButton {{
            background-color: {BG_SURFACE};
            border: 2px solid {accent};
            border-radius: {RADIUS}px;
            padding: 6px 16px;
            font-weight: 600;
            color: {accent};
        }}
        QPushButton:hover {{ background-color: {tint_rgba(accent, 0.10)}; }}
        QPushButton:pressed {{ background-color: {tint_rgba(accent, 0.18)}; }}
        QPushButton:disabled {{ background-color: {BG_MAIN}; border-color: #e5e7eb; color: {TEXT_DISABLED}; }}
    """


def build_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_MAIN))
    palette.setColor(QPalette.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(BG_SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(BG_MAIN))
    palette.setColor(QPalette.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(BG_SURFACE))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ToolTipBase, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_MUTED))
    palette.setColor(QPalette.Mid, QColor(BORDER))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DISABLED))
    return palette


def app_icon_pixmap(size: int) -> QPixmap:
    """A simple generated mark (rounded square + "L") at one exact pixel
    size — no external asset needed. Used directly wherever a crisp logo of
    a specific size is needed (e.g. the landing page); app_icon() below
    layers several of these into a QIcon for OS-level chrome (window/taskbar)
    instead, which picks and scales among a fixed set of sizes on its own."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(ACCENT))
    margin = max(1, round(size * 0.06))
    painter.drawRoundedRect(margin, margin, size - 2 * margin, size - 2 * margin, size * 0.22, size * 0.22)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(round(size * 0.58))
    painter.setFont(font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "L")
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    """A simple generated mark (rounded square + "L") — no external asset needed."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        icon.addPixmap(app_icon_pixmap(size))
    return icon


APP_STYLESHEET = f"""
QWidget {{
    background-color: {BG_MAIN};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY_CSS};
    font-size: {FS_BODY}px;
}}

QMainWindow, QDialog {{
    background-color: {BG_MAIN};
}}

QFrame#Card {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_CARD}px;
}}

QPushButton {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 7px 16px;
    color: {TEXT_PRIMARY};
    font-weight: 600;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
    color: {ACCENT_TEXT};
}}
QPushButton:pressed {{
    background-color: {ACCENT_SOFT_BG};
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    border-color: #e5e7eb;
    background-color: {BG_MAIN};
}}
QPushButton:checked {{
    border: 2px solid {ACCENT};
    background-color: {ACCENT_SOFT_BG};
}}

QToolBar {{
    background-color: {BG_SURFACE};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
    spacing: 6px;
}}
QToolButton {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 12px;
    color: {TEXT_PRIMARY};
    font-weight: 600;
}}
QToolButton:hover {{
    border: 1px solid {ACCENT};
    color: {ACCENT_TEXT};
}}
QToolButton:pressed {{
    background-color: {ACCENT_SOFT_BG};
}}
QToolButton:disabled {{
    color: {TEXT_DISABLED};
    border-color: #e5e7eb;
}}

QComboBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 20px;
}}
QComboBox:hover {{
    border: 1px solid {ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT_BG};
    selection-color: {TEXT_PRIMARY};
    outline: none;
}}

QLineEdit, QSpinBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
}}
QLineEdit:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}

QListView, QListWidget, QScrollArea, QGraphicsView, QTextBrowser {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {BG_MAIN};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 16px;
    margin-right: 2px;
    color: {TEXT_MUTED};
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background-color: {BG_SURFACE};
    color: {ACCENT_TEXT};
}}
QTabBar::tab:hover {{
    color: {ACCENT_TEXT};
}}

QProgressBar {{
    background-color: {BG_MAIN};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    font-weight: 600;
    font-size: {FS_CAPTION}px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

QStatusBar {{
    background-color: {BG_SURFACE};
    border-top: 1px solid {BORDER};
}}

QSplitter::handle {{
    background-color: {BG_MAIN};
}}
QSplitter::handle:hover {{
    background-color: {ACCENT};
}}

QMenu {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 7px 22px;
    border-radius: 5px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_SOFT_BG};
    color: {ACCENT_TEXT};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QLabel {{
    border: none;
}}

QToolTip {{
    background-color: {TEXT_PRIMARY};
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 5px 9px;
    font-size: {FS_CAPTION}px;
}}
"""
