import sys
from PyQt5.QtWidgets import QApplication
from glider_gui import GliderGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GliderGUI()
    window.show()
    sys.exit(app.exec_())