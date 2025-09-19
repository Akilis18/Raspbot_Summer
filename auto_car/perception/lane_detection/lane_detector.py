import os
import sys
import cv2
import numpy as np

# Add the absolute path to lane.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from lane import Lane

class LaneDetector:
    """
    車道檢測器 - 專為即時應用設計，保持狀態並重用計算
    """
    
    def __init__(self, frame_width, frame_height, plot_enabled=False):
        """
        初始化檢測器
        
        :param frame_width: 影像寬度
        :param frame_height: 影像高度
        :param plot_enabled: 是否啟用 plot 顯示
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lane_obj = None
        self.is_initialized = False
        self.plot_enabled = plot_enabled
        
        
        # 用於判斷是否使用滑動窗口的參數
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5

        # 用於移動平均濾波
        self.prev_left_fit = []
        self.prev_right_fit = []

        # 用於儲存最後一次成功的擬合結果
        self.last_left_fit = None
        self.last_right_fit = None

        self.LANE_WIDTH_PIXELS = 380  # 假設車道寬度為 380 像素(鳥瞰圖中)
        
    def initialize(self, first_frame):
        """
        使用第一幀初始化車道檢測器
        
        :param first_frame: 第一幀影像
        """
        # 調整影像大小（如果需要）
        if first_frame.shape[1] != self.frame_width or first_frame.shape[0] != self.frame_height:
            first_frame = cv2.resize(first_frame, (self.frame_width, self.frame_height))
        
        self.lane_obj = Lane(orig_frame=first_frame)
        
        # 執行初始檢測
        lane_line_markings = self.lane_obj.get_line_markings()
        warped_frame = self.lane_obj.perspective_transform()
        histogram = self.lane_obj.calculate_histogram(plot=self.plot_enabled)
        
        # 使用滑動窗口進行第一次檢測
        left_fit, right_fit = self.lane_obj.get_lane_line_indices_sliding_windows(plot=self.plot_enabled)
        
        if left_fit is not None and right_fit is not None:
            self.is_initialized = True
            self.frames_since_detection = 0
            # print("車道檢測器初始化成功")
        else:
            pass
            # print("車道檢測器初始化失敗")
            
        return self.is_initialized
    
    def enable_plot(self):
        """
        啟用 plot 顯示
        """
        self.plot_enabled = True
    
    def disable_plot(self):
        """
        關閉 plot 顯示
        """
        self.plot_enabled = False
    
    def set_plot(self, enabled):
        """
        設定 plot 顯示狀態
        
        :param enabled: True 為啟用，False 為關閉
        """
        self.plot_enabled = enabled
    
    def process_frame(self, frame, force_sliding_window=False, show_real_time=False):
        """
        處理單一幀影像（即時使用）
        
        :param frame: 輸入影像 (numpy.ndarray)
        :param force_sliding_window: 強制使用滑動窗口 (bool)
        :param show_real_time: 是否在回傳影像上繪製額外資訊 (bool)
        :return: 一個包含三個元素的元組 (tuple):
            - processed_frame (numpy.ndarray): 處理後的影像。
              如果檢測成功，則為帶有車道覆蓋和資訊的影像；
              如果失敗，則為原始影像。
            - success (bool): 如果成功檢測到車道線，則為 True，否則為 False。
            - lane_info (dict or None): 如果檢測成功，則為包含車道資訊的字典，否則為 None。
              字典包含以下鍵值:
                - 'left_curvature' (float): 左車道線的曲率半徑（米）。
                - 'right_curvature' (float): 右車道線的曲率半徑（米）。
                - 'center_offset' (float): 車輛中心相對於車道中心的偏移量（厘米）。
                - 'turn_direction' (str): 建議的轉向方向 ('left', 'right', 'straight')。
                - 'steer_deg' (float): 建議的轉向角度（度），左轉為正，右轉為負。
                - 'signed_radius_m' (float): 帶正負號的車道中心曲率半徑（米）。左轉為正，右轉為負，直線為 inf。
                - 'detection_method' (str): 使用的檢測方法 ('sliding_window' 或 'previous_window')。
                
        """
        if not self.is_initialized:
            success = self.initialize(frame)
            if not success:
                return frame, False, None
        
        # 調整影像大小
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # 更新 Lane object 的當前幀
        self.lane_obj.orig_frame = frame.copy()
        
        try:
            lane_line_markings = self.lane_obj.get_line_markings()
            self.lane_obj.perspective_transform()
            
            # 【修改點】決定使用哪種檢測方法的邏輯
            use_sliding_window = (
                force_sliding_window or 
                self.frames_since_detection >= self.max_frames_without_detection or
                self.last_left_fit is None or 
                self.last_right_fit is None
            )
            
            raw_left_fit, raw_right_fit = None, None
            detection_method = ''

            if use_sliding_window:
                self.lane_obj.calculate_histogram(plot=self.plot_enabled)
                raw_left_fit, raw_right_fit = self.lane_obj.get_lane_line_indices_sliding_windows(plot=self.plot_enabled)
                detection_method = 'sliding_window'
            else:
                raw_left_fit, raw_right_fit = self.lane_obj.get_lane_line_previous_window(
                    self.last_left_fit, self.last_right_fit, plot=self.plot_enabled
                )
                detection_method = 'previous_window'

            #在這裡進行後援與平滑化
            final_left_fit, final_right_fit = None, None

            # 情況一：同時偵測到左右兩條線
            if raw_left_fit is not None and raw_right_fit is not None:
                self.frames_since_detection = 0
                # 進行移動平均濾波
                self.prev_left_fit.append(raw_left_fit)
                self.prev_right_fit.append(raw_right_fit)
                if len(self.prev_left_fit) > 10:
                    self.prev_left_fit.pop(0)
                    self.prev_right_fit.pop(0)
                final_left_fit = sum(self.prev_left_fit) / len(self.prev_left_fit)
                final_right_fit = sum(self.prev_right_fit) / len(self.prev_right_fit)

            # 情況二：只偵測到左線
            elif raw_left_fit is not None and raw_right_fit is None:
                self.frames_since_detection = 0
                # 只對左線進行濾波
                self.prev_left_fit.append(raw_left_fit)
                if len(self.prev_left_fit) > 10:
                    self.prev_left_fit.pop(0)
                final_left_fit = sum(self.prev_left_fit) / len(self.prev_left_fit)

                # 根據左線建立虛擬的右線
                # A, B 係數相同 (曲率/斜率相同)，C 係數 (水平偏移) 不同
                final_right_fit = final_left_fit.copy()
                final_right_fit[2] += self.LANE_WIDTH_PIXELS

            # 情況三：只偵測到右線
            elif raw_left_fit is None and raw_right_fit is not None:
                self.frames_since_detection = 0
                # 只對右線進行濾波
                self.prev_right_fit.append(raw_right_fit)
                if len(self.prev_right_fit) > 10:
                    self.prev_right_fit.pop(0)
                final_right_fit = sum(self.prev_right_fit) / len(self.prev_right_fit)

                # 根據右線建立虛擬的左線
                final_left_fit = final_right_fit.copy()
                final_left_fit[2] -= self.LANE_WIDTH_PIXELS

            # 情況四：偵測完全失敗
            else:
                self.frames_since_detection += 1
                if self.last_left_fit is not None:
                    # 使用最後一次成功的結果作為後援
                    final_left_fit = self.last_left_fit
                    final_right_fit = self.last_right_fit
                else:
                    # 徹底失敗
                    return frame, False, None

            # 更新最後一次成功的結果
            self.last_left_fit = final_left_fit
            self.last_right_fit = final_right_fit

            # 將最終平滑化的結果賦值給 lane_obj 以便後續計算
            self.lane_obj.left_fit = final_left_fit
            self.lane_obj.right_fit = final_right_fit
            
            # 產生繪圖用的座標點
            ploty = self.lane_obj.ploty = np.linspace(0, self.lane_obj.warped_frame.shape[0] - 1, self.lane_obj.warped_frame.shape[0])
            self.lane_obj.left_fitx = final_left_fit[0] * ploty**2 + final_left_fit[1] * ploty + final_left_fit[2]
            self.lane_obj.right_fitx = final_right_fit[0] * ploty**2 + final_right_fit[1] * ploty + final_right_fit[2]

            # --- 後續的處理流程 (計算、顯示等) ---
            frame_with_lanes = self.lane_obj.overlay_lane_lines(plot=self.plot_enabled)
            # 確保有像素數據再計算曲率
            if self.lane_obj.leftx is not None and self.lane_obj.rightx is not None:
                self.lane_obj.calculate_curvature(print_to_terminal=False)
            self.lane_obj.calculate_car_position(print_to_terminal=False)
            
            if show_real_time:
                final_frame = self.lane_obj.display_curvature_offset(frame=frame_with_lanes, plot=False)
            else:
                final_frame = frame_with_lanes

                # 計算轉向建議
            direction, steer_deg, signed_R = self.lane_obj.compute_turn_command()
                
            # 準備車道資訊
            lane_info = {
                'left_curvature': self.lane_obj.left_curvem,
                'right_curvature': self.lane_obj.right_curvem,
                'center_offset': self.lane_obj.center_offset,
                'turn_direction': direction,
                'steer_deg': steer_deg,
                'signed_radius_m': signed_R,
                'detection_method': detection_method,
            }
                
            return final_frame, True, lane_info

                
        except Exception as e:
            print(f"車道檢測錯誤: {e}")
            self.frames_since_detection += 1
            return frame, False, None
    
    def reset(self):
        """
        重置檢測器狀態
        """
        self.lane_obj = None
        self.is_initialized = False
        self.frames_since_detection = 0
        print("車道檢測器已重置")
    
    def get_lane_center_point(self, y_position=None):
        """
        取得指定高度的車道中心點 X 座標
        
        :param y_position: Y座標位置（None為底部）
        :return: 車道中心點的 X 座標
        """
        if not self.is_initialized or self.lane_obj.left_fit is None or self.lane_obj.right_fit is None:
            return None
            
        if y_position is None:
            y_position = self.frame_height - 1
            
        # 計算該高度的左右車道線位置
        left_x = (self.lane_obj.left_fit[0] * y_position**2 + 
                 self.lane_obj.left_fit[1] * y_position + 
                 self.lane_obj.left_fit[2])
        right_x = (self.lane_obj.right_fit[0] * y_position**2 + 
                  self.lane_obj.right_fit[1] * y_position + 
                  self.lane_obj.right_fit[2])
        
        return (left_x + right_x) / 2
