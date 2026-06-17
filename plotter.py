"""
Arquivo da interface gráfica.
A interface foi feita utilizando QT6.

Existem 3 telas
- Um emulador de terminal simples.
- Um plotter dos erros
- Uma galeria dos dados de treinamento, 
que pode ser acessado em View/Data

Cada uma das telas são classes diferentes. 
Idealmente sendo auto-contidas mas com acesso imediato as variáveis
e funções disponíveis em core.py.
"""

import pyqtgraph as pg

from PySide6.QtCore import (
    QObject, Qt, Signal, Slot
)

from PySide6.QtGui import (
    QAction, QImage, QPixmap
)

from PySide6.QtWidgets import (
    QApplication, QDockWidget, QMainWindow, QGridLayout,
    QPlainTextEdit, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QDialog, QSplitter
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
    compartilhadas da aplicação, todos os widgets importantes tem acesso 
    a uma única instância dessa classe.
    """
    _training: bool

    training_changed = Signal(bool)
    next_gen = Signal(None)

    start_training: QAction

    def __init__(self):
        super().__init__()
        self._training = False

        self.start_training = QAction("Treinar", self)
        self.start_training.triggered.connect(self.train)
        
        # cada geração essa função é chamada
        core.on_next_gen = self._on_next_gen

    def _on_next_gen(self):
        self.next_gen.emit()

    def train(self):
        def do_training(): 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(core.train())
            finally:
                self.training = False;
                loop.close()

        if self.training == True:
            return;

        print("INICIANDO NOVO TREINAMENTO")

        self.training = True;
        threading.Thread(target=do_training, daemon=True).start()

    @property
    def training(self) -> bool:
        return self._training
    
    @training.setter
    def training(self, value: bool):
        self._training = value
        self.training_changed.emit(value)

class DataWidget(QWidget):
    """
    Um widget que mostra um único dado de treinamento.
    """
    def __init__(self, data: core.training_data):
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

    
    def training2image(self, data: core.training_data) -> QImage:
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

class DataView(QScrollArea):
    """
    A view que mostra todos os dados de treinamento.
    """
    def __init__(self, state: AppState):
        super().__init__()

        self.setWidgetResizable(False)  # IMPORTANT for fixed grid
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        grid = QGridLayout(container)
        grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        grid.setSpacing(5)

        cols = 3

        for i, d in enumerate(core.data):
            row = i // cols
            col = i % cols

            w = DataWidget(d)
            w.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

            grid.addWidget(w, row, col)

        container.adjustSize()
        self.setWidget(container)

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
        sys.stderr = self.stdout_redirector

    def append_log(self, text: str):
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


class MainView(QMainWindow):
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

        menu_bar.addAction(self.state.start_training)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        #
        # plot widget
        #
        self.plot_widget = PlotView(self.state)
        self.splitter.addWidget(self.plot_widget)

        #
        # Log dock
        #
        self.log_widget = LogView(self.state)

        self.splitter.addWidget(self.log_widget)

        #
        # data dock
        #
        self.data_dock = QDockWidget("Data", self)
        self.data_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | 
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.data_dock.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
        
        self.data_dock.setFloating(True)
        self.data_dock.setWindowFlags(Qt.WindowType.Window) 

        self.data_widget = DataView(self.state)
        self.data_dock.setWidget(self.data_widget)

        self.data_dock.show()
        self.data_dock.toggleViewAction().trigger()

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.data_dock.toggleViewAction())

def main():
    app = QApplication()
    qt_themes.set_theme('monokai')

    main_wind = MainView()
    main_wind.setWindowTitle("Plotter")

    main_wind.show()
    app.exec()

if __name__ == "__main__":
    main()
