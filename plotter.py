"""
Arquivo da interface gráfica.
A interface foi feita utilizando QT6.

Existem 3 views
- Um emulador de terminal simples.
- Um plotter dos erros
- Uma galeria dos dados de treinamento, que pode ser acessado em View/Data


Cada uma das telas são classes diferentes.
Idealmente sendo auto-contidas mas com acesso imediato as variáveis
e funções disponíveis em core.py.
"""

import pyqtgraph as pg
import config

from PySide6.QtCore import (
    QObject, Qt, Signal, Slot
)

from pathlib import Path

from PySide6.QtGui import (
    QAction, QImage, QPixmap, QTextCursor
)

from PySide6.QtWidgets import (
    QApplication, QDockWidget, QMainWindow, QGridLayout, QLineEdit,
    QPlainTextEdit, QWidget, QVBoxLayout, QLabel, QScrollArea, QSpinBox, QDoubleSpinBox,
    QSizePolicy, QHBoxLayout, QFormLayout, QPushButton, QFileDialog,
    QDialog
)

import threading
import core
import asyncio
import sys
import core
import numpy as np

import qt_themes

class SignalingStream(QObject):
    """
    Eu basicamente estou redirecionado o stdout do terminal para
    a interface gráfica.

    para sobrescrever o stdio, é necessário criar uma classe
    que implementa a função write e flush.
    e sobreescrever o objeto sys.stdout

    sys.stdout = novo_stdout
    """
    text_written = Signal(str)

    def write(self, text: str):
        if text:
            self.text_written.emit(text)

    def flush(self):
        pass


class AppState(QObject):
    """
    Classe utilitária que implementa algumas funcionalidades básicas
    compartilhadas da aplicação, todos as views importantes tem acesso 
    a uma única instância dessa classe.
    """
    _working: bool

    working_changed = Signal(bool)
    next_gen = Signal(None)
    training_data_reloaded = Signal(list)
    config_reloaded = Signal(None)
    testing_data_reloaded = Signal(list)

    train_action: QAction
    test_action: QAction
    reload_training_data_action: QAction
    reload_testing_data_action: QAction
    store_config_action: QAction
    reload_config_action: QAction
    open_config_action: QAction

    def __init__(self):
        super().__init__()
        self._working = False

        self.train_action = QAction("Treinar", self)
        self.train_action.triggered.connect(self.train)

        self.test_action = QAction("Testar", self)
        self.test_action.triggered.connect(self.test)

        self.reload_training_data_action = QAction("Recarregar dados de treinamento")
        self.reload_training_data_action.triggered.connect(self.reload_training_data)

        self.reload_testing_data_action = QAction("Recarregar dados de testes")
        self.reload_testing_data_action.triggered.connect(self.reload_working_data)

        self.store_config_action = QAction("Salvar configuração")
        self.store_config_action.triggered.connect(config.store_config)

        self.load_config_action = QAction("Carregar configuração")
        self.load_config_action.triggered.connect(self.reload_config)
        self.config_reloaded.connect(self.reload_training_data_action.trigger)

        self.open_config_action = QAction("Configurar")
        self.open_config_action.triggered.connect(self.open_config_dialog)

        # cada geração essa função é chamada
        core.on_next_gen = self._on_next_gen

    def reload_training_data(self):
        core.load_training_data()
        self.training_data_reloaded.emit(core.training_data)
    
    def reload_working_data(self):
        core.load_testing_data()
        self.testing_data_reloaded.emit(core.testing_data)

    def store_config(self):
        config.store_config()

    def reload_config(self):
        config.load_config()
        self.config_reloaded.emit()

    def _on_next_gen(self):
        self.next_gen.emit()

    def open_config_dialog(self):
        dialog = QDialog()
        dialog.setWindowTitle("Configuração")

        layout = QVBoxLayout(dialog)
        layout.addWidget(ConfigView(self))

        dialog.exec()

    def train(self):
        self._do(core.train)

    def test(self):
        self._do(core.test_working_data)

    def _do(self, fn):
        def async_fn(): 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(fn())
            finally:
                self.working = False;
                loop.close()

        if self.working == True:
            return;

        self.working = True;
        threading.Thread(target=async_fn, daemon=True).start()

    @property
    def working(self) -> bool:
        return self._working
    
    @working.setter
    def working(self, value: bool):
        self._working = value
        self.working_changed.emit(value)

class DataWidget(QWidget):
    """
    Um widget que mostra um único dado de treinamento.
    """
    def __init__(self, data: core.manuscript_image):
        super().__init__()
        self.setFixedSize(130, 150)
        self.setMinimumSize(0,0)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.icon = QLabel()

        self.icon.setPixmap(QPixmap.fromImage(self.training2image(data)))

        self.text = QLabel()
        self.text.setText(str(data.y))
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon)
        layout.addWidget(self.text)

    
    def training2image(self, data: core.manuscript_image) -> QImage:
        """
        Essa função é responsável por pegar um dado de treinamento
        e converter para uma imagem.
        """
        m = np.array([
            data.x[1:4],
            data.x[4:7],
            data.x[7:10]
        ])
        pixels = np.where(m == -1, 0, 255).astype(np.uint8)
        h, w = pixels.shape

        img = QImage(pixels.data, w, h, pixels.strides[0],
                     QImage.Format.Format_Grayscale8)
        
        return img.scaled(w*40, h*40, Qt.KeepAspectRatio, Qt.FastTransformation).copy()

class DataGalleryView(QScrollArea):
    """
    A view que mostra todos os dados de treinamento.
    """
    state: AppState

    def __init__(self, state: AppState, data):
        super().__init__()
        self.state = state

        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.grid = QGridLayout(container)
        self.grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        self.grid.setSpacing(5)

        self.load_data(data)

        container.adjustSize()
        self.setWidget(container)

    def load_data(self, data):
        self.data = data
        while True:
            item = self.grid.takeAt(0)
            if item is None:
                break

            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        cols = 3

        for i, d in enumerate(self.data):
            row = i // cols
            col = i % cols

            w = DataWidget(d)
            self.grid.addWidget(w, row, col)

class FilesystemSelectWidget(QWidget):
    pathChanged = Signal(str)

    def __init__(self, initial_path=""):
        super().__init__()

        self.edit = QLineEdit(initial_path)
        self.button = QPushButton("Browse...")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # or 4
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

        self.button.clicked.connect(self.browse)
        self.edit.textChanged.connect(self.pathChanged)

    @property
    def path(self):
        return self.edit.text()

    @path.setter
    def path(self, value):
        self.edit.setText(value)

    def browse(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            self.path,
        )

        if path:
            self.path = path

class TrainingDataView(QWidget):
    state: AppState
    def __init__(self, state):
        super().__init__()
        self.state = state

        self.gallery = DataGalleryView(state, core.training_data);
        self.gallery.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.state.training_data_reloaded.connect(self.gallery.load_data)

        self.config = QWidget()
        reload_button = QPushButton()
        reload_button.setText(self.state.reload_training_data_action.text())
        reload_button.clicked.connect(self.state.reload_training_data_action.trigger)
        train_button = QPushButton()
        train_button.setText(self.state.train_action.text())
        train_button.clicked.connect(self.state.train_action.trigger)

        select_data = FilesystemSelectWidget(str(config.c_training_data_path))
        def change_path(path: str):
            config.c_training_data_path = Path(path);
        select_data.pathChanged.connect(change_path)

        config_layout = QVBoxLayout()
        self.config.setLayout(config_layout)
        config_layout.addWidget(select_data)
        config_layout.addWidget(reload_button)
        config_layout.addWidget(train_button)

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)
        main_layout.addWidget(self.gallery)
        main_layout.addWidget(self.config)

class TestingDataView(QWidget):
    state: AppState
    def __init__(self, state):
        super().__init__()
        self.state = state

        self.gallery = DataGalleryView(state, core.testing_data);
        self.gallery.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.state.testing_data_reloaded.connect(self.gallery.load_data)

        self.config = QWidget()
        reload_button = QPushButton()
        reload_button.setText(self.state.reload_testing_data_action.text())
        reload_button.clicked.connect(self.state.reload_testing_data_action.trigger)
        test_button = QPushButton()
        test_button.setText(self.state.test_action.text())
        test_button.clicked.connect(self.state.test_action.trigger)

        select_data = FilesystemSelectWidget(str(config.c_testing_data_path))
        def change_path(path: str):
            config.c_testing_data_path = Path(path);
        select_data.pathChanged.connect(change_path)

        config_layout = QVBoxLayout()
        self.config.setLayout(config_layout)
        config_layout.addWidget(select_data)
        config_layout.addWidget(reload_button)
        config_layout.addWidget(test_button)

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)
        main_layout.addWidget(self.gallery)
        main_layout.addWidget(self.config)

class LogView(QPlainTextEdit):
    """
    A view que loga o texto do terminal
    """
    state: AppState

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setReadOnly(True)

        self.stdout_redirector = SignalingStream()
        self.stdout_redirector.text_written.connect(self.append_log)
        
        sys.stdout = self.stdout_redirector
        #sys.stderr = self.stdout_redirector


    def append_log(self, text: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

        self.insertPlainText(text)
        self.ensureCursorVisible()

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

class PlotView(QWidget):
    """
    A view que plota o erro ao decorrer do tempo
    """
    state: AppState

    def __init__(self, state: AppState):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)

        self.plot.setLabel("left", "Erro")
        self.plot.setLabel("bottom", "Geração")

        self.curve = self.plot.plot([], [], pen="y")

        layout.addWidget(self.plot)

        self.x_data = []
        self.y_data = []
        vb = self.plot.getViewBox()
        if vb.state['autoRange'][0]:
            vb.enableAutoRange()

        self.state.next_gen.connect(self.on_next_gen)

    @Slot()
    def on_next_gen(self):
        self.y_data = core.errors
        self.x_data = range(len(core.errors))

        self.curve.setData(self.x_data, self.y_data)


class ConfigView(QWidget):
    state: AppState
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.config_buffer = config.__dict__.copy()

        def save_config():
            config.__dict__.clear()
            config.__dict__.update(self.config_buffer)
            self.state.store_config_action.trigger()

        self.apply_button = QPushButton("salvar")
        self.apply_button.clicked.connect(save_config)

        options_container = QWidget()
        options_layout = QHBoxLayout()
        options_container.setLayout(options_layout)
        options_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        options_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        options_layout.addWidget(self.apply_button)
        
        config_scroll = QScrollArea()
        config_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        config_scroll.setWidgetResizable(True)

        config_container = QWidget()
        config_scroll.setWidget(config_container)
        config_form = QFormLayout()
        config_container.setLayout(config_form)

        for item in self.config_buffer.items():
            if not item[0].startswith("c_"):
                continue
            widget = None
            key = item[0]

            def update_generic(value, key=key):
                self.config_buffer[key] = value

            if isinstance(item[1], Path):
                widget = FilesystemSelectWidget(str(item[1]))
                def change_path(path, key=key):
                    self.config_buffer[key] = Path(path)

                widget.pathChanged.connect(change_path);

            elif isinstance(item[1], str):
                widget = QLineEdit(item[1])
                widget.setText(item[1])
                widget.textChanged.connect(update_generic)

            elif isinstance(item[1], int):
                widget = QSpinBox()
                widget.setValue(item[1])
                widget.valueChanged.connect(update_generic)

            elif isinstance(item[1], float):
                widget = QDoubleSpinBox()
                widget.setValue(item[1])
                widget.valueChanged.connect(update_generic)

            config_form.addRow(item[0][2:], widget)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(config_scroll)
        main_layout.addWidget(options_container)

class MainWindow(QMainWindow):
    """
    Janela principal, nada de especial.
    """
    def __init__(self):
        super().__init__()
    
        self.state = AppState()
    
        #
        # Menu bar
        #
        menu_bar = self.menuBar()
        #menu_bar.addAction(self.state.train_action)
        menu_bar.addAction(self.state.open_config_action)
    
        #
        # plot widget (central)
        #
        self.plot_widget = PlotView(self.state)
        self.setCentralWidget(self.plot_widget)
    
        #
        # Log dock
        #
        self.log_widget = LogView(self.state)
    
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.log_dock
        )
    
        #
        # training data dock
        #
        self.training_data_dock = QDockWidget("Dados de treino", self)
        self.training_data_dock.setWidget(TrainingDataView(self.state))

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.training_data_dock
        )

        #
        # working data dock
        #
        self.testing_data_dock = QDockWidget("Dados de teste", self)
        self.testing_data_dock.setWidget(TestingDataView(self.state))

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.testing_data_dock
        )

        self.tabifyDockWidget(
            self.training_data_dock,
            self.testing_data_dock
        )

        self.training_data_dock.raise_()
    
        #
        # View menu
        #
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addAction(self.training_data_dock.toggleViewAction())
        view_menu.addAction(self.testing_data_dock.toggleViewAction())
    
        self.setDockNestingEnabled(True)

def main():
    app = QApplication()
    qt_themes.set_theme('monokai')

    main_wind = MainWindow()
    main_wind.setWindowTitle("Plotter")

    main_wind.show()
    app.exec()

if __name__ == "__main__":
    main()
