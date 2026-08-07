"""In-app help: a tabbed guide explaining how to use LabelMaker.

Kept as static HTML strings rendered in QTextBrowser rather than a pile of
QLabels — simpler to write, and QTextBrowser handles wrapping/scrolling
correctly on its own (unlike QLabel + QScrollArea, which has bitten this
app's layout before).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QTextBrowser, QVBoxLayout

from app.shared.theme import ACCENT, FS_BODY, FS_H1, TASK_ACCENTS, TEXT_MUTED, TEXT_PRIMARY, primary_button_style

_BODY_STYLE = f"""
    body {{ color: {TEXT_PRIMARY}; font-size: {FS_BODY}px; line-height: 155%; }}
    h2 {{ color: {ACCENT}; font-size: {FS_H1}px; margin-bottom: 2px; }}
    h3 {{ color: {TEXT_PRIMARY}; font-size: {FS_BODY}px; margin-top: 18px; margin-bottom: 2px; }}
    p {{ margin-top: 4px; color: {TEXT_PRIMARY}; }}
    ul, ol {{ margin-top: 4px; padding-left: 20px; }}
    li {{ margin-bottom: 5px; }}
    .lede {{ color: {TEXT_MUTED}; }}
"""


def _wrap(html: str) -> str:
    return f"<html><head><style>{_BODY_STYLE}</style></head><body>{html}</body></html>"


def _color_icon(hex_color: str, size: int = 12) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(hex_color))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)


def _browser(html: str) -> QTextBrowser:
    browser = QTextBrowser()
    browser.setOpenExternalLinks(False)
    browser.setHtml(_wrap(html))
    return browser


_OVERVIEW_HTML = """
<h2>Welcome to LabelMaker</h2>
<p class="lede">A desktop tool for preparing labeled image datasets for computer vision.</p>
<p>LabelMaker supports three different kinds of labeling:</p>
<ul>
<li><b>Classification</b> — tag each whole image with one class (e.g. "this photo is a dog").</li>
<li><b>Detection</b> — draw a box around each object in an image (a dog <i>and</i> a cat in the same photo each get their own box).</li>
<li><b>Segmentation</b> — trace a precise outline around each object instead of a rectangle.</li>
</ul>
<h3>The basic flow</h3>
<ol>
<li>From the home screen, pick a task type, or open a project you already started.</li>
<li>Import images — a whole folder, or individual files.</li>
<li>Add the classes you want to label with (e.g. Dog, Cat, Car).</li>
<li>Label your images — the exact steps depend on the task; see the other tabs above.</li>
<li>For Detection, export the finished dataset in a training-ready format.</li>
</ol>
<h3>Good to know</h3>
<ul>
<li><b>Nothing is lost silently.</b> Every change saves to disk immediately — there's no Save button, and nothing to remember before closing the app.</li>
<li><b>Your original files are safe.</b> Labeling never modifies or moves your images, except Classification's optional "Move" mode, which you turn on deliberately.</li>
<li>Click <b>Home</b> in the top bar any time to leave a project and start or open another.</li>
</ul>
"""

_CLASSIFICATION_HTML = """
<h2>Classification</h2>
<p class="lede">One class per image.</p>
<p>Use this when each photo, as a whole, belongs to a single category.</p>
<h3>1. Import your images</h3>
<p><b>Import Folder</b> brings in every image in a folder (and its subfolders); <b>Import Image</b> adds specific files. <b>Import Classes</b> loads a plain text file with one class name per line, instead of typing them in one by one.</p>
<h3>2. Add your classes</h3>
<p>Click <b>+ Add Class</b> and give it a name, color, and an optional single-key shortcut. Each class becomes a button — click one to make it the <i>active</i> class (only one is active at a time; it's the one new labels go to).</p>
<h3>3. Label your images</h3>
<p>Select one or more images in the grid (click, or Ctrl/Shift-click for several), make sure the class you want is active, then click <b>Assign (Enter)</b> — or just press Enter. The selected images are labeled immediately and drop out of the "Unlabeled" queue.</p>
<h3>Copy vs. Move</h3>
<p>The <b>Mode</b> toggle next to Assign controls what happens to the file: <b>Copy</b> (default) places a copy into a folder named after the class and leaves your original alone; <b>Move</b> relocates the original file into that folder instead.</p>
<h3>Filtering</h3>
<p>The <b>Filter</b> dropdown above the grid switches between <i>Unlabeled</i> (your remaining work), <i>All</i>, and <i>By Class</i> (review what's already assigned to one class).</p>
"""

_DETECTION_HTML = """
<h2>Detection</h2>
<p class="lede">Bounding boxes around objects.</p>
<p>Use this to locate individual objects within a photo — one image can contain several objects of different classes, each gets its own box.</p>
<h3>1. Import images &amp; add classes</h3>
<p>Same as Classification — <b>Import Folder/Image</b> and <b>+ Add Class</b> in the left panel.</p>
<h3>2. Draw a box</h3>
<p>Click a class in the <b>Classes</b> panel to make it active — this only decides the class of the <i>next</i> box you draw. Then click and drag on the image to draw a box around an object.</p>
<h3>3. Edit a box</h3>
<p>Click an existing box to select it: drag its body to move it, drag a corner or edge to resize it, press <b>Delete</b> to remove it.</p>
<h3>Multiple objects, multiple classes</h3>
<p>Every box keeps whatever class it was drawn with — switching the active class only affects new boxes, not existing ones. The <b>Objects in this image</b> panel lists every box on the current image; use it to select a box on the canvas, reassign its class without redrawing, or delete it.</p>
<h3>Exporting your dataset</h3>
<p>Once you've labeled enough images, click <b>Export Dataset</b> (top right) to save everything in a training-ready format:</p>
<ul>
<li><b>YOLO (.txt)</b> — images/labels split into train/val/test folders, plus a classes.yaml.</li>
<li><b>Pascal VOC (.xml)</b> — one XML file per image, plus train/val/test image-list files.</li>
<li><b>CreateML (.json)</b> — one images-plus-annotations.json folder per split.</li>
</ul>
<p>You choose the train/val/test percentages and an output folder; exporting only writes new files there — your project itself is untouched.</p>
"""

_SEGMENTATION_HTML = """
<h2>Segmentation</h2>
<p class="lede">Precise outlines around objects.</p>
<p>Like Detection, but instead of a rectangle you trace the object's actual outline as a polygon — useful when a box would include too much background, or the exact shape matters.</p>
<h3>1. Import images &amp; add classes</h3>
<p>Same as the other tasks — <b>Import Folder/Image</b> and <b>+ Add Class</b>.</p>
<h3>2. Draw a shape</h3>
<p>Click a class to make it active, then click once for each point around the object's edge — a corner, a curve, wherever the outline changes direction. Keep clicking your way around the whole object; each point is numbered as you go.</p>
<h3>3. Close the shape</h3>
<p>Finish it any of three ways: click back on the very first point (it turns green once you're close enough), press <b>Enter</b>, or double-click. You need at least 3 points before a shape can close.</p>
<h3>4. Edit a shape</h3>
<p>Click a shape to select it: drag its body to move the whole thing, drag any point to reshape it, double-click a point to delete just that point (as long as at least 3 remain). Press <b>Delete</b> to remove the whole selected shape, or <b>Escape</b> to cancel one you're still drawing.</p>
<h3>Multiple objects, multiple classes</h3>
<p>Just like Detection, every shape keeps its own class no matter what's active now, and the <b>Objects in this image</b> panel lets you select, reassign, or delete any of them.</p>
"""

_TIPS_HTML = """
<h2>Tips &amp; keyboard shortcuts</h2>
<h3>Navigation</h3>
<ul>
<li><b>F</b> — fit the current image to the view.</li>
<li>Mouse wheel over an image — zoom in/out (the +/− and <b>Fit</b> buttons do the same).</li>
<li><b>A</b> / <b>D</b> — jump to the previous / next image in the grid.</li>
<li>Digit keys (<b>1</b>–<b>9</b>) — instantly activate a class, if you gave it that shortcut when creating it.</li>
</ul>
<h3>Working with images</h3>
<ul>
<li>Click an image to select it; Ctrl-click or Shift-click to select several at once.</li>
<li><b>Delete</b>/<b>Backspace</b>, or right-click → <i>Remove from Project</i> — removes the selected image(s) from the project (asks first). This never deletes the file on disk, it only stops tracking it here.</li>
</ul>
<h3>Working with classes</h3>
<ul>
<li>Right-click a class button to <b>Edit</b> its name/color/shortcut, or <b>Remove</b> it. Removing a class asks for confirmation — it clears that class from every image/object that used it, and can't be undone.</li>
</ul>
<h3>Detection &amp; Segmentation</h3>
<ul>
<li><b>Delete</b> — removes the currently selected box/shape.</li>
<li><b>Escape</b> — cancels a segmentation shape you're still drawing.</li>
</ul>
<h3>Saving</h3>
<p>Everything autosaves the instant you make a change — there's no Save button and no confirmation dialog when closing. There also isn't a separate undo history, so undoing a mistake means redoing it by hand (move the box back, redraw the shape, reassign the class).</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help — LabelMaker")
        self.resize(700, 580)

        tabs = QTabWidget()
        tabs.addTab(_browser(_OVERVIEW_HTML), "Overview")
        tabs.addTab(_browser(_CLASSIFICATION_HTML), _color_icon(TASK_ACCENTS["classification"]), "Classification")
        tabs.addTab(_browser(_DETECTION_HTML), _color_icon(TASK_ACCENTS["detection"]), "Detection")
        tabs.addTab(_browser(_SEGMENTATION_HTML), _color_icon(TASK_ACCENTS["segmentation"]), "Segmentation")
        tabs.addTab(_browser(_TIPS_HTML), "Tips && Shortcuts")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).setStyleSheet(primary_button_style(ACCENT))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(tabs, stretch=1)
        layout.addWidget(buttons)
