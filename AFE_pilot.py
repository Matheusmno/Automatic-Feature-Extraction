from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import * 
from PyQt5.QtCore import Qt
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 1.5 * self.WINDOW_WIDTH
  
        # this will hide the title bar
        #self.setWindowFlag(Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground)
        
        # setting  the geometry of window
        self.setGeometry(100, 100, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setWindowTitle("Automatic Feature Extraction")
        #self.showFullScreen()

        # calling method
        self.UIComponents()
  
        # showing all the widgets
        self.show()
    
    # method for widgets
    def UIComponents(self):
        def read_directory_path(self):
            dialog = QtWidgets.QFileDialog()
            folder_path = dialog.getExistingDirectory(None, "Select directory containing the EDF input files")
            return folder_path
    
        def addPathButton(self):
            path_button = QtWidgets.QPushButton("Load EDF files", self)
            path_button.setGeometry(100, 200 , 250, 100)
            path_button.clicked.connect(read_directory_path)
        
        addPathButton(self)

if __name__ == "__main__":
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    # start the app
    sys.exit(App.exec())