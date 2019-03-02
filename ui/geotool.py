# Date: 2019.02.26
# Author: Kingdrone
from PyQt5.QtWidgets import *
import sys
from ui.gdal_tool_ui import Ui_MainWindow
from src.utils import tif_process, tif_merge
import cgitb
from src.utils import judge_util
from src.utils.event_register import ThreadWrapper, ProcessHandler, reset_log
cgitb.enable(format = "text")
import os
import glob


class Demo(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("GeoTool")
        self.setMinimumSize(0, 0)
        clip_type_list = ['格网裁剪', '矢量裁剪']

        self.OPEN_TIF_ACTION.triggered.connect(self.open_tif)
        self.OPEN_SHAPEFILE_ACTION.triggered.connect(self.open_shape_file)
        self.OPEN_MERGE_BN.clicked.connect(self.open_dir)
        self.OPEN_SAVE_BN.clicked.connect(self.open_save_dir)
        self.OPEN_GEOREFED_BN.clicked.connect(self.open_georefed_dir)
        self.OPEN_MASK_BN.clicked.connect(self.open_mask_dir)
        self.RUN_REF_BN.clicked.connect(self.run_define_reference)
        self.CLIP_TYPE_COMBOX.addItems(clip_type_list)
        self.RUN_CLIP_BN.clicked.connect(self.run_clip_tif)
        self.RUN_MASK_BN.clicked.connect(self.run_mask_tif)
        self.RUN_MERGE_BN.clicked.connect(self.run_merge)

        # process handler init
        self.process_handler = ProcessHandler()
        # 连接log信号
        self.process_handler.process_signal.connect(self.update_process)
        # properties
        self.tif = None
        self.shapefile = None
        self.shapefile_path = ""
        self.save_dir = ""
        self.refered_tif_dir=""
        self.mask_dir=""
        self.tif_list =[]

    def open_dir(self):
        tif_dir = QFileDialog.getExistingDirectory(self, "选取保存路径", "/home")
        tif_list = glob.glob(os.path.join(tif_dir, "*.tif"))
        if len(tif_list) > 0:
            self.tif_list = tif_list
            self.MERGE_LINEEDIT.setText(tif_dir)
            QMessageBox.information(self, "读入成功！", "已成功读入%d个TIF"%len(tif_list))
        else:
            QMessageBox.information(self, "读入失败！", "批量读入TIF失败！请检查文件内容！")

    def open_tif(self):
        open_tif_path, _ = QFileDialog.getOpenFileName(self,
                                                      "选取TIF文件",
                                                      "/home","tif(*.tif)")  # 起始路径
        try:
            self.tif = tif_process.GeoTiff(open_tif_path)
            bands_count = self.tif.bands_count
            lng, lat  = self.tif.get_left_top()
            pixel_h, pixel_w = self.tif.get_pixel_height_width()
            tif_col, tif_row = self.tif.col, self.tif.row

            tif_txt_info = "路径：%s\n波段数：%s\n左上角坐标：(%.3f，%.3f)\n" \
                           "像素高宽：%.3e,%.3e\n像素行列数：%d,%d"%(
                open_tif_path, int(bands_count),float(lng), float(lat), float(pixel_h), float(pixel_w),
                int(tif_row), int(tif_col)
            )
            if open_tif_path != "":
                self.TIF_TXTEDIT.setText(tif_txt_info)
        except Exception:
            QMessageBox.information(self,  # 使用infomation信息框
                                            "错误",
                                            "文件格式错误！")
    def open_shape_file(self):
        open_shapefile_path, _ = QFileDialog.getOpenFileName(self,
                                                       "选取shapfile文件",
                                                       "/home","shapefile(*.shp)")  # 起始路径
        self.shapefile_path = open_shapefile_path
        try:
            self.shapefile = tif_process.GeoShaplefile(open_shapefile_path)
            ft = self.shapefile.feature_type
            minX, minY, maxX, maxY = self.shapefile.minX, self.shapefile.minY, self.shapefile.maxX, self.shapefile.maxY
            num_ft = self.shapefile.feature_num
            shpfile_txt_info = "路径：%s\n要素类型：%s\n要素数量：%d\n图层范围：(%.3f，%.3f，%.3f，%.3f)"\
                           %(self.shapefile_path, ft, num_ft, minX, minY, maxX, maxY)
            if self.shapefile_path != "":
                self.SHAPEFILE_TXTEDIT.setText(shpfile_txt_info)
        except Exception:
            QMessageBox.information(self,  # 使用infomation信息框
                                            "错误",
                                            "文件格式错误！")
    def open_save_dir(self):
        self.save_dir = QFileDialog.getExistingDirectory(self, "选取保存路径", "/home")
        self.SAVE_PATH_DIR_LINEEDIT.setText(self.save_dir)
    def open_georefed_dir(self):
        self.refered_tif_dir = QFileDialog.getExistingDirectory(self, "选取保存路径", "/home")
        self.GEOREFED_LINEEDIT.setText(self.refered_tif_dir)
    def open_mask_dir(self):
        self.mask_dir = QFileDialog.getExistingDirectory(self, "选取保存路径", "/home")
        self.UNGEOREFED_LINEEDIT.setText(self.mask_dir)
    def run_clip_tif(self):
        reset_log()
        if self.check_tif() and self.check_savedir():
            # clip with grid
            if self.CLIP_TYPE_COMBOX.currentIndex() == 0:
                if not self.check_clipsize():
                    return
                clip_size = self.CLIP_SIZE_LINEEDIT.text()
                self.process_handler.start()

                tw = ThreadWrapper(self.tif.clip_tif_with_grid,
                                   int(clip_size), self.save_dir)
                tw.start()

            # clip with shapefile
            if self.CLIP_TYPE_COMBOX.currentIndex() == 1:
                if not self.check_shp():
                    QMessageBox.information(self, "错误", "尚未读取shapefile！")
                    return
                self.process_handler.start()
                tw = ThreadWrapper(self.tif.clip_tif_with_shapefile,
                                   self.shapefile_path, self.save_dir)
                tw.start()

        return
    def run_mask_tif(self):
        reset_log()
        if self.check_tif() and self.check_savedir() \
            and self.check_shp() and self.check_mask_label():

            mask_label = int(self.MASK_LABEL_LINEEDIT.text())
            save_path = os.path.join(self.save_dir, 'mask.tif')
            self.process_handler.start()
            tw = ThreadWrapper(self.tif.mask_tif_with_shapefile,
                               self.shapefile_path, save_path, mask_label)
            tw.start()

    def run_define_reference(self):
        reset_log()
        img_dir = self.refered_tif_dir
        mask_dir = self.mask_dir
        if img_dir == "":
            QMessageBox.information(self, "错误", "请选择无坐标系分类结果路径！")
            return
        if mask_dir == "":
            QMessageBox.information(self, "错误", "请选择有坐标系影像路径！")
            return
        if len(os.listdir(img_dir)) != len(os.listdir(mask_dir)):
            QMessageBox.information(self, "错误", "分类结果与影像数量不一致！")
            return
        if self.check_savedir():
            self.process_handler.start()
            tw = ThreadWrapper(tif_process.define_ref_predict,
                               img_dir, mask_dir, self.save_dir)
            tw.start()
    def run_merge(self):
        reset_log()
        if len(self.tif_list) == 0:
            QMessageBox.information(self, "错误", "尚未批量读取TIF！")
            return
        if self.check_savedir():
            save_path = os.path.join(self.save_dir, "merge.tif")
            self.process_handler.start()
            tw = ThreadWrapper(tif_merge.run_merge,
                               self.tif_list, save_path)
            tw.start()


    def check_clipsize(self):
        clip_size = self.CLIP_SIZE_LINEEDIT.text()
        if not judge_util.is_number(clip_size) or clip_size == "":
            QMessageBox.information(self, "错误", "请填写正确的裁剪尺寸！")
            return False
        else:
            return True

    def check_mask_label(self):
        mask_label = self.MASK_LABEL_LINEEDIT.text()
        if not judge_util.is_number(mask_label) or mask_label == "":
            QMessageBox.information(self, "错误", "请填写正确的腌膜数值！")
            return False
        else:
            return True

    def check_tif(self):
        if self.tif is None:
            QMessageBox.information(self, "错误", "尚未打开TIF文件")
            return False
        return True

    def check_shp(self):
        if self.shapefile is None:
            QMessageBox.information(self, "错误", "尚未读取shapefile！")
            return False
        return True

    def check_savedir(self):
        if self.save_dir == "":
            QMessageBox.information(self, "错误", "请填写保存路径！")
            return False
        return True

    def update_process(self, s):
        self.PROCESS_BAR.setValue(s)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Demo()
    main.show()
    sys.exit(app.exec_())
